/*===------- llvm/Config/llvm-config-unreachable-optimize.h.cmake -*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_CONFIG_UNREACHABLE_OPTIMIZE_H
#define LLVM_CONFIG_UNREACHABLE_OPTIMIZE_H

/* Define if llvm_unreachable should be optimized with undefined behavior
 * in non assert builds */
#cmakedefine01 LLVM_UNREACHABLE_OPTIMIZE

#endif // LLVM_CONFIG_UNREACHABLE_OPTIMIZE_H
