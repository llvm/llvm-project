// RUN: %libomp-compile && env OMP_THREAD_LIMIT=4 %libomp-run 4
// RUN: %libomp-compile && env OMP_THREAD_LIMIT=7 %libomp-run 7
//
// OMP_THREAD_LIMIT=N should imply that no more than N threads are active in
// a contention group
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "omp_testsuite.h"

int failed = 0;

void usage() {
    fprintf(stderr, "usage: omp_thread_limit <n>\n");
}

void verify(const char* file_name, int line_number, int team_size) {
  int num_threads = omp_get_num_threads();
  if (team_size != num_threads) {
#pragma omp critical(A)
    {
      char label[256];
      snprintf(label, sizeof(label), "%s:%d", file_name, line_number);
      failed = 1;
      printf("failed: %s: team_size(%d) != omp_get_num_threads(%d)\n",
             label, team_size, num_threads);
    }
  }
}

int main(int argc, char** argv)
{
  int cl_thread_limit;

  if (argc != 2) {
    usage();
    return 1;
  }
  cl_thread_limit = atoi(argv[1]);

  omp_set_dynamic(0);
  if (omp_get_thread_limit() != cl_thread_limit) {
    fprintf(stderr, "omp_get_thread_limit failed with %d, should be%d\n",
            omp_get_thread_limit(), cl_thread_limit);
    return 1;
  }
  else if (omp_get_max_threads() > cl_thread_limit) {
#if _OPENMP
    int team_size = cl_thread_limit;
#else
    int team_size = 1;
#endif
    omp_set_num_threads(19);
    verify(__FILE__, __LINE__, 1);
#pragma omp parallel
    {
      verify(__FILE__, __LINE__, team_size);
      verify(__FILE__, __LINE__, team_size);
    }
    verify(__FILE__, __LINE__, 1);

    omp_set_nested(1);
#pragma omp parallel num_threads(3)
    {
      verify(__FILE__, __LINE__, 3);
#pragma omp master
#pragma omp parallel num_threads(21)
      {
        verify(__FILE__, __LINE__, team_size-2);
        verify(__FILE__, __LINE__, team_size-2);
      }
    }
    verify(__FILE__, __LINE__, 1);

    return failed;
  } else {
    fprintf(stderr, "This test is not applicable for max num_threads='%d'\n",
            omp_get_max_threads());
    return 0;
  }

}
