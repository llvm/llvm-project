// The OpenMP standard defines 3 ways of providing ompt_start_tool:
// 1. "statically-linking the tool’s definition of ompt_start_tool into an OpenMP application"
// RUN: %libomp-compile -DCODE -DTOOL && %libomp-run | FileCheck %s

// Note: We should compile the tool without -fopenmp as other tools developer
//       would do. Otherwise this test may pass for the wrong reasons on Darwin.
// RUN: %clang %flags -DTOOL -shared -fPIC %s -o %T/tool.so
// 2. "introducing a dynamically-linked library that includes the tool’s definition of ompt_start_tool into the application’s address space"
// 2.1 Link with tool during compilation
// RUN: %libomp-compile -DCODE %T/tool.so && %libomp-run | FileCheck %s
// 2.2 Link with tool during compilation, but AFTER the runtime
// RUN: %libomp-compile -DCODE -lomp %T/tool.so && %libomp-run | FileCheck %s
// 2.3 Inject tool via the dynamic loader
// RUN: %libomp-compile -DCODE && %preload-tool %libomp-run | FileCheck %s

// 3. "providing the name of a dynamically-linked library appropriate for the architecture and operating system used by the application in the tool-libraries-var ICV"
// RUN: %libomp-compile -DCODE && env OMP_TOOL_LIBRARIES=%T/tool.so %libomp-run | FileCheck %s

// REQUIRES: ompt

/*
 *  This file contains code for an OMPT shared library tool to be
 *  loaded and the code for the OpenMP executable.
 *  -DTOOL enables the code for the tool during compilation
 *  -DCODE enables the code for the executable during compilation
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
  printf("0: ompt_event_runtime_shutdown\n");
}

ompt_start_tool_result_t* ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,&ompt_finalize, 0};
  return &ompt_start_tool_result;
}
#endif /* TOOL */
