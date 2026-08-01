/*===-- Compiler.h - Compiler abstractions for the ORC RT C API ---*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Compiler-abstraction macros used by the ORC runtime C API headers.         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_COMPILER_H
#define ORC_RT_C_COMPILER_H

/* Helper to promote strict prototype warnings to errors */
#ifdef __clang__
#define ORC_RT_C_STRICT_PROTOTYPES_BEGIN                                       \
  _Pragma("clang diagnostic push")                                             \
      _Pragma("clang diagnostic error \"-Wstrict-prototypes\"")
#define ORC_RT_C_STRICT_PROTOTYPES_END _Pragma("clang diagnostic pop")
#else
#define ORC_RT_C_STRICT_PROTOTYPES_BEGIN
#define ORC_RT_C_STRICT_PROTOTYPES_END
#endif

/* Helper to wrap C code for C++ */
#ifdef __cplusplus
#define ORC_RT_C_EXTERN_C_BEGIN                                                \
  extern "C" {                                                                 \
  ORC_RT_C_STRICT_PROTOTYPES_BEGIN
#define ORC_RT_C_EXTERN_C_END                                                  \
  ORC_RT_C_STRICT_PROTOTYPES_END                                               \
  }
#else
#define ORC_RT_C_EXTERN_C_BEGIN ORC_RT_C_STRICT_PROTOTYPES_BEGIN
#define ORC_RT_C_EXTERN_C_END ORC_RT_C_STRICT_PROTOTYPES_END
#endif

/* ORC_RT_C_ABI is the export/visibility macro used to mark symbols declared
   in orc-rt-c as exported when built as a shared library. */
#if defined(__has_attribute) && __has_attribute(visibility)
#define ORC_RT_C_ABI __attribute__((visibility("default")))
#endif

#if !defined(ORC_RT_C_ABI)
#define ORC_RT_C_ABI
#endif

/* ORC_RT_C_NOTHROW indicates that a function won't throw a C++ exception. */
#if defined(__cplusplus)
#define ORC_RT_C_NOTHROW noexcept
#elif defined(__GNUC__) || defined(__clang__)
#define ORC_RT_C_NOTHROW __attribute__((nothrow))
#else
#define ORC_RT_C_NOTHROW
#endif

/* ORC_RT_C_FORMAT_PRINTF(FmtIdx, FirstArg) marks a function as taking a
   printf-style format string at argument position FmtIdx, with the variadic
   arguments beginning at FirstArg, so the compiler can check the format string
   against its arguments. Indices are 1-based. */
#if defined(__GNUC__) || defined(__clang__)
#define ORC_RT_C_FORMAT_PRINTF(FmtIdx, FirstArg)                               \
  __attribute__((format(printf, FmtIdx, FirstArg)))
#else
#define ORC_RT_C_FORMAT_PRINTF(FmtIdx, FirstArg)
#endif

#endif /* ORC_RT_C_COMPILER_H */
