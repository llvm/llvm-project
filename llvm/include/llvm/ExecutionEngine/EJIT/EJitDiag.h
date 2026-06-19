//===-- EJitDiag.h - EmbeddedJIT Diagnostic Logging ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostic logging for the EmbeddedJIT runtime, used for bring-up,
// field debugging, and production monitoring.  All output goes through
// the platform-provided SRE_printf() so it integrates with existing
// device log infrastructure.
//
// Compile with -DEJIT_DIAG_ENABLE to activate logging.  When not defined,
// all macros expand to nothing and incur zero runtime cost.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITDIAG_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITDIAG_H

#ifdef EJIT_DIAG_ENABLE

// Provided by the platform.  Must be declared exactly as written so the
// linker can resolve it against the device firmware's libc / BSP.
extern "C" int SRE_printf(const char *fmt, ...);

// Split the prefix, payload, and newline into separate calls so the macro works
// in C++17 without the GNU `, ##__VA_ARGS__` extension. Diagnostics are a
// bring-up-only facility, so preserving one atomic printf call is not required.
#define EJIT_DIAG(...)                                                     \
  do {                                                                     \
    SRE_printf("[EJIT] %s:%d ", __func__, __LINE__);                       \
    SRE_printf(__VA_ARGS__);                                               \
    SRE_printf("\n");                                                      \
  } while (0)

#else // !EJIT_DIAG_ENABLE

// Expand to ((void)0) regardless of argument count by matching everything.
#define EJIT_DIAG(...) ((void)0)

#endif // EJIT_DIAG_ENABLE

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITDIAG_H
