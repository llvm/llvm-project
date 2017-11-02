// RUN: %libomp-compile-and-run
#include <omp.h>

int main()
{
  #pragma omp parallel num_threads(1)
  {
    omp_control_tool(omp_control_tool_flush, 1, NULL);
  }

  return 0;
}
