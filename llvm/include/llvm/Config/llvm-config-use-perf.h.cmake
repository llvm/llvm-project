/*===------- llvm/Config/llvm-config-use-perf.h.cmake -------------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_CONFIG_USE_PERF_H
#define LLVM_CONFIG_USE_PERF_H

/* Define if we have the perf JIT-support library */
#cmakedefine01 LLVM_USE_PERF

#endif // LLVM_CONFIG_USE_PERF_H
