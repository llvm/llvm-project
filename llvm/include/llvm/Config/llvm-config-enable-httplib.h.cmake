/*===------- llvm/Config/llvm-config-httplib.h --------------------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_CONFIG_ENABLE_HTTPLIB_H
#define LLVM_CONFIG_ENABLE_HTTPLIB_H

/* Define if we have cpp-httplib and want to use it */
#cmakedefine LLVM_ENABLE_HTTPLIB ${LLVM_ENABLE_HTTPLIB}

#endif // LLVM_CONFIG_ENABLE_HTTPLIB_H
