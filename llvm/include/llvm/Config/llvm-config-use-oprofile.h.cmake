/*===------- llvm/Config/llvm-config-oprofile.h -------------------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_CONFIG_USE_OPROFILE_H
#define LLVM_CONFIG_USE_OPROFILE_H

/* Define if we have the oprofile JIT-support library */
#cmakedefine01 LLVM_USE_OPROFILE

#endif // LLVM_CONFIG_USE_OPROFILE_H
