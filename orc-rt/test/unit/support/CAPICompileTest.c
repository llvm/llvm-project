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
