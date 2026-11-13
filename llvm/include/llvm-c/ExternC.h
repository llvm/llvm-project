/*===- llvm-c/ExternC.h - Wrapper for 'extern "C"' ----------------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file defines an 'extern "C"' wrapper                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_EXTERNC_H
#define LLVM_C_EXTERNC_H

#ifdef __clang__
#define LLVM_C_STRICT_PROTOTYPES_BEGIN                                         \
  _Pragma("clang diagnostic push")                                             \
      _Pragma("clang diagnostic error \"-Wstrict-prototypes\"")
#define LLVM_C_STRICT_PROTOTYPES_END _Pragma("clang diagnostic pop")
#else
#define LLVM_C_STRICT_PROTOTYPES_BEGIN
#define LLVM_C_STRICT_PROTOTYPES_END
#endif

#ifdef __cplusplus
#define LLVM_C_EXTERN_C_BEGIN                                                  \
  extern "C" {                                                                 \
  LLVM_C_STRICT_PROTOTYPES_BEGIN
#define LLVM_C_EXTERN_C_END                                                    \
  LLVM_C_STRICT_PROTOTYPES_END                                                 \
  }
#else
#define LLVM_C_EXTERN_C_BEGIN LLVM_C_STRICT_PROTOTYPES_BEGIN
#define LLVM_C_EXTERN_C_END LLVM_C_STRICT_PROTOTYPES_END
#endif

#ifndef LLVM_HAS_ATTRIBUTE
#if defined(__has_attribute)
#define LLVM_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define LLVM_HAS_ATTRIBUTE(x) 0
#endif
#endif

#if LLVM_HAS_ATTRIBUTE(warn_unused_result) && defined(__clang__)
#define LLVM_NODISCARD_TYPEDEF __attribute__((warn_unused_result))
#else
#define LLVM_NODISCARD_TYPEDEF
#endif

#endif
