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

#ifndef ORC_RT_C_SUPPORT_COMPILER_H
#define ORC_RT_C_SUPPORT_COMPILER_H

/* ORC_RT_HAS_BUILTIN(X) expands to 1 if the compiler provides the builtin X,
   and to 0 otherwise.

   This wraps __has_builtin rather than supplying a fallback definition for it.
   __has_builtin is a reserved identifier, and defining one from a public header
   can collide with the compiler's own definition or with other libraries that
   the client also includes. */
#if defined(__has_builtin)
#define ORC_RT_HAS_BUILTIN(X) __has_builtin(X)
#else
#define ORC_RT_HAS_BUILTIN(X) 0
#endif

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

/* ORC_RT_C_EXPORT marks a symbol declared in orc-rt-c as part of the ORC
   runtime's binary interface: exported from the runtime when it is built as a
   shared library, and imported by consumers of that library.

   TODO: Add the Windows __declspec(dllexport) / __declspec(dllimport) and
   static-build cases once there is a shared-library build to exercise them. */
#if defined(__has_attribute) && __has_attribute(visibility)
#define ORC_RT_C_EXPORT __attribute__((visibility("default")))
#endif

#if !defined(ORC_RT_C_EXPORT)
#define ORC_RT_C_EXPORT
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

#endif /* ORC_RT_C_SUPPORT_COMPILER_H */
