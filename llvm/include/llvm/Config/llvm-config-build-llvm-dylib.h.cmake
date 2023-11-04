/*===------- llvm/Config/llvm-config-build-llvm-dylib.h.cmake -----*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_CONFIG_BUILD_LLVM_DYLIB_H
#define LLVM_CONFIG_BUILD_LLVM_DYLIB_H

/* Define if building libLLVM shared library */
#cmakedefine LLVM_BUILD_LLVM_DYLIB ${LLVM_BUILD_LLVM_DYLIB}

#endif // LLVM_CONFIG_BUILD_LLVM_DYLIB_H
