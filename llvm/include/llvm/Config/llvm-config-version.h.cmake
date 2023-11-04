/*===------- llvm/Config/llvm-config-version.h - llvm config ------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_CONFIG_VERSION_H
#define LLVM_CONFIG_VERSION_H

/* This file defines macros for the LLVM version. */

/* Major version of the LLVM API */
#define LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR}

/* Minor version of the LLVM API */
#define LLVM_VERSION_MINOR ${LLVM_VERSION_MINOR}

/* Patch version of the LLVM API */
#define LLVM_VERSION_PATCH ${LLVM_VERSION_PATCH}

/* LLVM version string */
#define LLVM_VERSION_STRING "${PACKAGE_VERSION}"

#endif // LLVM_CONFIG_VERSION_H
