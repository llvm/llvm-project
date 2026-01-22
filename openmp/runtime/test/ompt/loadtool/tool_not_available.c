// clang-format off
// The OpenMP standard defines 3 ways of providing ompt_start_tool:

// RUN: mkdir -p %t.tool_dir

// 1. "statically-linking the tool’s definition of ompt_start_tool into an
// OpenMP application"

// RUN: %libomp-compile -DCODE -DTOOL && \
// RUN:    env OMP_TOOL_VERBOSE_INIT=stdout %libomp-run | \
// RUN:    FileCheck %s --check-prefixes CHECK,ADDRSPACE 

// Note: We should compile the tool without -fopenmp as other tools developer
//      would do. Otherwise this test may pass for the wrong reasons on Darwin.

// RUN: %clang %flags -DTOOL -shared -fPIC %s -o %t.tool_dir/tool.so

// 2. "introducing a dynamically-linked library that includes the tool’s 
//    definition of ompt_start_tool into the application’s address space"

// 2.1 Link with tool during compilation

// RUN: %libomp-compile -DCODE %no-as-needed-flag %t.tool_dir/tool.so && \
// RUN:    env OMP_TOOL_VERBOSE_INIT=stdout %libomp-run | \
// RUN:    FileCheck %s --check-prefixes CHECK,ADDRSPACE

// 2.2 Link with tool during compilation, but AFTER the runtime

// RUN: %libomp-compile -DCODE -lomp %no-as-needed-flag %t.tool_dir/tool.so && \
// RUN:    env OMP_TOOL_VERBOSE_INIT=stdout %libomp-run | \
// RUN:    FileCheck %s --check-prefixes CHECK,ADDRSPACE 

// 2.3 Inject tool via the dynamic loader

// RUN: %libomp-compile -DCODE && \
// RUN:    env OMP_TOOL_VERBOSE_INIT=stdout %preload-tool %libomp-run | \
// RUN:    FileCheck %s --check-prefixes CHECK,ADDRSPACE 

// 3. "providing the name of a dynamically-linked library appropriate for the
//    architecture and operating system used by the application in the 
//    tool-libraries-var ICV"

// RUN: %libomp-compile -DCODE && env OMP_TOOL_LIBRARIES=%t.tool_dir/tool.so \
// RUN:    OMP_TOOL_VERBOSE_INIT=stdout %libomp-run | \
// RUN:    FileCheck %s -DPARENTPATH=%t.tool_dir --check-prefixes CHECK,TOOLLIB

// REQUIRES: ompt
// clang-format on

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

int main() {
#pragma omp parallel num_threads(2)
  {
#pragma omp master
    {
      int result = omp_control_tool(omp_control_tool_start, 0, NULL);
      printf("0: control_tool()=%d\n", result);
    }
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback

  // ADDRSPACE: ----- START LOGGING OF TOOL REGISTRATION -----
  // ADDRSPACE-NEXT: Search for OMP tool in current address space...

  // TOOLLIB: ----- START LOGGING OF TOOL REGISTRATION -----
  // TOOLLIB-NEXT: Search for OMP tool in current address space... Failed.
  // TOOLLIB-NEXT: Searching tool libraries...
  // TOOLLIB-NEXT: OMP_TOOL_LIBRARIES = [[PARENTPATH]]/tool.so
  // TOOLLIB-NEXT: Opening [[PARENTPATH]]/tool.so... Success.
  // TOOLLIB-NEXT: Searching for ompt_start_tool in
  // TOOLLIB-SAME: [[PARENTPATH]]/tool.so...

  // CHECK: 0: Do not initialize tool

  // ADDRSPACE-NEXT: Failed.
  // ADDRSPACE-NEXT: No OMP_TOOL_LIBRARIES defined.
  // ADDRSPACE-NEXT: ...searching tool libraries failed.
  // ADDRSPACE: No OMP tool loaded.
  // ADDRSPACE-NEXT: ----- END LOGGING OF TOOL REGISTRATION -----

  // TOOLLIB-NEXT: Found but not using the OMPT interface.
  // TOOLLIB-NEXT: Continuing search...
  // TOOLLIB-NEXT: ...searching tool libraries failed.
  // TOOLLIB: No OMP tool loaded.
  // TOOLLIB-NEXT: ----- END LOGGING OF TOOL REGISTRATION -----

  // CHECK: {{^}}0: control_tool()=-2
  // clang-format on

  return 0;
}

#endif /* CODE */

#ifdef TOOL

#include <omp-tools.h>
#include "stdio.h"

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  printf("0: Do not initialize tool\n");
  return NULL;
}
#endif /* TOOL */
