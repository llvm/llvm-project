/*===--- Visibility.h - Visibility macros for the ORC runtime ---*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header defines visibility macros used for the ORC runtime C interface.*|
|* These macros are used to annotate C functions that should be exported as   *|
|* part of a shared library or DLL.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_VISIBILITY_H
#define ORC_RT_C_VISIBILITY_H

/* ORC_RT_C_ABI is the export/visibility macro used to mark symbols declared
   in orc-rt-c as exported when built as a shared library. */

#if defined(__has_attribute) && __has_attribute(visibility)
#define ORC_RT_C_ABI __attribute__((visibility("default")))
#endif

#if !defined(ORC_RT_C_ABI)
#define ORC_RT_C_ABI
#endif

#endif /* ORC_RT_C_VISIBILITY_H */
