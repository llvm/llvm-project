/*===------- llvm/Config/llvm-config-target-triple.h --------------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

/* This file defines macros about the targets configuration. */

#ifndef LLVM_CONFIG_TARGET_TRIPLE_H
#define LLVM_CONFIG_TARGET_TRIPLE_H

/* Target triple LLVM will generate code for by default */
/* Doesn't use `cmakedefine` because it is allowed to be empty. */
#define LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}"

/* Host triple LLVM will be executed on */
#cmakedefine LLVM_HOST_TRIPLE "${LLVM_HOST_TRIPLE}"

#endif // LLVM_CONFIG_TARGET_TRIPLE_H