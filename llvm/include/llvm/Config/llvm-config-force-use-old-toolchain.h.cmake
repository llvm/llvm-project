/*===------- llvm/Config/llvm-config-force-use-old-toolchain.h ----*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_CONFIG_FORCE_USE_OLD_TOOLCHAIN_H
#define LLVM_CONFIG_FORCE_USE_OLD_TOOLCHAIN_H

/* Define if building LLVM with LLVM_FORCE_USE_OLD_TOOLCHAIN_LIBS */
#cmakedefine LLVM_FORCE_USE_OLD_TOOLCHAIN ${LLVM_FORCE_USE_OLD_TOOLCHAIN}

#endif // LLVM_CONFIG_FORCE_USE_OLD_TOOLCHAIN_H
