// RUN: %libomp-compile -DCODE && %libomp-compile -DTOOL -o%T/tool.so -shared -fPIC && env OMP_TOOL_LIBRARIES=%T/tool.so %libomp-run | FileCheck %s
// REQUIRES: ompt

/*
 *  This file contains code for an OMPT shared library tool to be 
 *  loaded and the code for the OpenMP executable. 
 *  -DTOOL enables the code for the tool during compilation
 *  -DCODE enables the code for the executable during compilation
 *  The RUN line compiles the two binaries and then tries to load
 *  the tool using the OMP_TOOL_LIBRARIES environmental variable.
 */

#ifdef CODE
#include "omp.h"

int main()
{
  #pragma omp parallel num_threads(2)
  {
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 
  
  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}0: ompt_event_runtime_shutdown

  return 0;
}

#endif /* CODE */

#ifdef TOOL

#include <stdio.h>
#include <ompt.h>

int ompt_initialize(
  ompt_function_lookup_t lookup,
  ompt_data_t* tool_data)
{
  printf("0: NULL_POINTER=%p\n", (void*)NULL);
  return 1; //success
}

void ompt_finalize(ompt_data_t* tool_data)
{
  printf("%d: ompt_event_runtime_shutdown\n", omp_get_thread_num());
}

ompt_start_tool_result_t* ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,&ompt_finalize, 0};
  return &ompt_start_tool_result;
}
#endif /* TOOL */
