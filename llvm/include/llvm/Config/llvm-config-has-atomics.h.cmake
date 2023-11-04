/*===------- llvm/Config/llvm-config-has-atomics.h.cmake ----------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_CONFIG_HAS_ATOMICS_H
#define LLVM_CONFIG_HAS_ATOMICS_H

/* Has gcc/MSVC atomic intrinsics */
#cmakedefine01 LLVM_HAS_ATOMICS

#endif // LLVM_CONFIG_HAS_ATOMICS_H
