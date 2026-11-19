/*===- orc-rt-test-dylib.c - Shared library for testing -----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* A minimal shared library, for tests that load, search, and close a host    *|
|* library. Each function returns a distinct value, so that a test can check  *|
|* that a lookup found the function it asked for.                             *|
|*                                                                            *|
|* The functions are found by lookup (e.g. dlsym), not by linking against the *|
|* library, so there is no header to declare them.                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(_WIN32)
#define ORC_RT_TEST_DYLIB_EXPORT __declspec(dllexport)
#else
#define ORC_RT_TEST_DYLIB_EXPORT __attribute__((visibility("default")))
#endif

ORC_RT_TEST_DYLIB_EXPORT int orc_rt_test_dylib_return42(void) { return 42; }
ORC_RT_TEST_DYLIB_EXPORT int orc_rt_test_dylib_return7(void) { return 7; }
