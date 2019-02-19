// The OpenMP standard defines 3 ways of providing ompt_start_tool:
// 1. "statically-linking the tool’s definition of ompt_start_tool into an OpenMP application"
// RUN: %libomp-compile -DCODE -DTOOL && %libomp-run | FileCheck %s

// Note: We should compile the tool without -fopenmp as other tools developer
//       would do. Otherwise this test may pass for the wrong reasons on Darwin.
// RUN: %clang %flags -DTOOL -shared -fPIC %s -o %T/tool.so
// 2. "introducing a dynamically-linked library that includes the tool’s definition of ompt_start_tool into the application’s address space"
// 2.1 Link with tool during compilation
// RUN: %libomp-compile -DCODE %no-as-needed-flag %T/tool.so && %libomp-run | FileCheck %s
// 2.2 Link with tool during compilation, but AFTER the runtime
// RUN: %libomp-compile -DCODE -lomp %no-as-needed-flag %T/tool.so && %libomp-run | FileCheck %s
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
#include "stdio.h"
#include "omp.h"
#include "omp-tools.h"

int main()
{
  #pragma omp parallel num_threads(2)
  {
    #pragma omp master
    {
      int result = omp_control_tool(omp_control_tool_start, 0, NULL);
      printf("0: control_tool()=%d\n", result);
    }
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 
  
  // CHECK: {{^}}0: Do not initialize tool
  // CHECK: {{^}}0: control_tool()=-2
  

  return 0;
}

#endif /* CODE */

#ifdef TOOL

#include <omp-tools.h>
#include "stdio.h"

ompt_start_tool_result_t* ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  printf("0: Do not initialize tool\n");
  return NULL;
}
#endif /* TOOL */
