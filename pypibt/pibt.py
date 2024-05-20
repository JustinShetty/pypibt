import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Configs, Coord, Grid, get_neighbors


class PIBT:
    def __init__(self, grid: Grid, starts: Config, goals: Config, seed: int = 0):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.N = len(self.starts)

        # distance table
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def funcPIBT(
        self,
        Q_from: Config,
        Q_to: Config,
        i: int,
        j: list[int],
    ) -> bool:
        # true -> valid, false -> invalid
        print(f"funcPIBT({i}, {j[-1] if j[-1] != self.NIL else 'NIL'})")

        # get candidate next vertices
        C = [Q_from[i]] if j[-1] == self.NIL else []
        C += get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: self.dist_tables[i].get(u))

        print(i, Q_from[i], C)

        # vertex assignment
        for v in C:
            print(i, v)

            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                print(i, f"vertex collision {self.occupied_nxt[v]}")
                continue

            # avoid following conflict
            k = self.occupied_now[v]
            if k != self.NIL and k != i:
                if Q_to[k] == self.NIL_COORD and k not in j:
                    print(i, "try priority inheritance", k)
                    Q_to[i] = Q_from[i]
                    self.occupied_nxt[Q_from[i]] = i
                    if self.funcPIBT(Q_from, Q_to, k, j + [i]):
                        print(i, "priority inheritance success")
                        return True
                    Q_to[i] = self.NIL_COORD
                    self.occupied_nxt[Q_from[i]] = self.NIL
                    print(i, "priority inheritance failed")
                continue

            print(i, "reserve next location")
            Q_to[i] = v
            self.occupied_nxt[v] = i
            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(self, Q_from: Config, priorities: list[float]) -> Config:
        # setup
        N = len(Q_from)
        Q_to: Config = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, i, [self.NIL])

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        assert np.all(self.occupied_now == self.NIL)
        if not np.all(self.occupied_nxt == self.NIL):
            with np.printoptions(threshold=np.inf):
                print(self.occupied_nxt)
            print(Q_from)
            print(Q_to)
            print(priorities)
            assert False

        return Q_to

    def run(self, max_timestep: int = 1000) -> Configs:
        # define priorities
        priorities: list[float] = []
        for i in range(self.N):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / self.grid.size)

        # main loop, generate sequence of configurations
        configs = [self.starts]
        while len(configs) <= max_timestep:
            # obtain new configuration
            print(f"t={len(configs)-1}")
            Q = self.step(configs[-1], priorities)
            print()
            configs.append(Q)

            # update priorities & goal check
            flg_fin = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += 1
                else:
                    priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break  # goal

        return configs
