/*===- CAPICompileTest.c - Check that the C API compiles as C -----*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Compiles the public C API headers as C, and exercises the macros in        *|
|* orc-rt-c/support/Compiler.h from C.                                        *|
|*                                                                            *|
|* The companion assertions are in CompilerTest.cpp, which calls the          *|
|* functions defined here.                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "orc-rt-c/bedrock/Session.h"
#include "orc-rt-c/support/Compiler.h"
#include "orc-rt-c/support/CoreTypes.h"
#include "orc-rt-c/support/Error.h"
#include "orc-rt-c/support/Logging.h"
#include "orc-rt-c/support/WrapperFunction.h"

/* ORC_RT_HAS_BUILTIN must be usable in a preprocessor conditional in C, and
   must agree with the answer the C++ compiler gives for the same builtin. */
#if ORC_RT_HAS_BUILTIN(__builtin_expect)
int orc_rt_test_hasBuiltinExpect(void) { return 1; }
#else
int orc_rt_test_hasBuiltinExpect(void) { return 0; }
#endif

/* A builtin no compiler provides must report 0 rather than failing to
   preprocess. */
#if ORC_RT_HAS_BUILTIN(__builtin_orc_rt_not_a_real_builtin)
#error "ORC_RT_HAS_BUILTIN reported support for a nonexistent builtin"
#endif

/* ORC_RT_LIKELY / ORC_RT_UNLIKELY must normalize their operand without relying
   on bool, which is not a keyword in C before C23. */
int orc_rt_test_likely(int X) { return ORC_RT_LIKELY(X) ? 1 : 0; }
int orc_rt_test_unlikely(int X) { return ORC_RT_UNLIKELY(X) ? 1 : 0; }

/* ORC_RT_WEAK_IMPORT must be applicable to a declaration in C. Never called;
   this only checks that the attribute parses here. */
ORC_RT_WEAK_IMPORT void orc_rt_test_weakImportDecl(void);

/* ORC_RT_UNREACHABLE must compile in a value-returning C function, and must
   satisfy the compiler that control does not fall off the end. */
int orc_rt_test_unreachable(int X) {
  switch (X) {
  case 0:
    return 0;
  default:
    ORC_RT_UNREACHABLE("only 0 is expected");
  }
}
