// REQUIRES: ompx_taskgraph
// RUN: %libomp-cxx-compile-and-run
#include <omp.h>
#include <cassert>
#include <vector>

constexpr const int TASKS_SIZE = 12;

typedef struct ident ident_t;

extern "C" {
int __kmpc_global_thread_num(ident_t *);
int __kmpc_start_record_task(ident_t *, int, int, int);
void __kmpc_end_record_task(ident_t *, int, int, int);
}

void init(int &A, int val) { A = val; }

void update(int &A, int &B, int val) { A = B + val; }

void test(int nb, std::vector<std::vector<int>> &Ah) {
#pragma omp parallel
#pragma omp single
  {
    int gtid = __kmpc_global_thread_num(nullptr);
    int res = __kmpc_start_record_task(nullptr, gtid, 0, 0);
    if (res) {
      for (int k = 0; k < nb; ++k) {
#pragma omp task depend(inout : Ah[k][0])
        init(Ah[k][0], k);

        for (int i = 1; i < nb; ++i) {
#pragma omp task depend(in : Ah[k][0]) depend(out : Ah[k][i])
          update(Ah[k][i], Ah[k][0], 1);
        }
      }
    }
    __kmpc_end_record_task(nullptr, gtid, 0, 0);
  }
}

int main() {
  std::vector<std::vector<int>> matrix(TASKS_SIZE,
                                       std::vector<int>(TASKS_SIZE, 0));

  test(TASKS_SIZE, matrix);
  test(TASKS_SIZE, matrix);

  for (int k = 0; k < TASKS_SIZE; ++k) {
    assert(matrix[k][0] == k);
    for (int i = 1; i < TASKS_SIZE; ++i) {
      assert(matrix[k][i] == k + 1);
    }
  }
  return 0;
}
