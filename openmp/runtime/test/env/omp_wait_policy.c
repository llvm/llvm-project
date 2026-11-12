// RUN: %libomp-compile && env OMP_WAIT_POLICY=active %libomp-run active
// RUN: %libomp-compile && env OMP_WAIT_POLICY=passive %libomp-run passive
//
// OMP_WAIT_POLICY=active should imply blocktime == INT_MAX
// i.e., threads spin-wait forever
// OMP_WAIT_POLICY=passive should imply blocktime == 0
// i.e., threads immediately sleep
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "omp_testsuite.h"

void usage() {
    fprintf(stderr, "usage: omp_wait_policy active|passive\n");
}

int main(int argc, char** argv)
{
  int blocktime, retval=1;
  const char* env_var_value;

  if (argc != 2) {
    usage();
    return 1;
  }

  blocktime = kmp_get_blocktime();

  env_var_value = argv[1];
  if (!strcmp(env_var_value, "active")) {
    retval = (blocktime != INT_MAX);
  } else if (!strcmp(env_var_value, "passive")) {
    retval = (blocktime != 0);
  } else {
    usage();
    retval = 1;
  }

  return retval;
}
