/*===------- llvm/Config/llvm-config.h - llvm configuration -------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

/* This file enumerates variables from the LLVM configuration so that they
   can be in exported headers and won't override package specific directives.
   This is a C header that can be included in the llvm-c headers. */

#ifndef LLVM_CONFIG_H
#define LLVM_CONFIG_H

/* Indicate that this is LLVM compiled from the amd-gfx branch. */
#define LLVM_HAVE_BRANCH_AMD_GFX
#define LLVM_MAIN_REVISION 430784

/* Define if LLVM_ENABLE_DUMP is enabled */
#cmakedefine LLVM_ENABLE_DUMP

/* Target triple LLVM will generate code for by default */
/* Doesn't use `cmakedefine` because it is allowed to be empty. */
#define LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}"

/* Define if threads enabled */
#cmakedefine01 LLVM_ENABLE_THREADS

/* Has gcc/MSVC atomic intrinsics */
#cmakedefine01 LLVM_HAS_ATOMICS

/* Host triple LLVM will be executed on */
#cmakedefine LLVM_HOST_TRIPLE "${LLVM_HOST_TRIPLE}"

/* LLVM architecture name for the native architecture, if available */
#cmakedefine LLVM_NATIVE_ARCH ${LLVM_NATIVE_ARCH}

/* LLVM name for the native AsmParser init function, if available */
#cmakedefine LLVM_NATIVE_ASMPARSER LLVMInitialize${LLVM_NATIVE_ARCH}AsmParser

/* LLVM name for the native AsmPrinter init function, if available */
#cmakedefine LLVM_NATIVE_ASMPRINTER LLVMInitialize${LLVM_NATIVE_ARCH}AsmPrinter

/* LLVM name for the native Disassembler init function, if available */
#cmakedefine LLVM_NATIVE_DISASSEMBLER LLVMInitialize${LLVM_NATIVE_ARCH}Disassembler

/* LLVM name for the native Target init function, if available */
#cmakedefine LLVM_NATIVE_TARGET LLVMInitialize${LLVM_NATIVE_ARCH}Target

/* LLVM name for the native TargetInfo init function, if available */
#cmakedefine LLVM_NATIVE_TARGETINFO LLVMInitialize${LLVM_NATIVE_ARCH}TargetInfo

/* LLVM name for the native target MC init function, if available */
#cmakedefine LLVM_NATIVE_TARGETMC LLVMInitialize${LLVM_NATIVE_ARCH}TargetMC

/* LLVM name for the native target MCA init function, if available */
#cmakedefine LLVM_NATIVE_TARGETMCA LLVMInitialize${LLVM_NATIVE_ARCH}TargetMCA

/* Define if this is Unixish platform */
#cmakedefine LLVM_ON_UNIX ${LLVM_ON_UNIX}

/* Define if we have the Intel JIT API runtime support library */
#cmakedefine01 LLVM_USE_INTEL_JITEVENTS

/* Define if we have the oprofile JIT-support library */
#cmakedefine01 LLVM_USE_OPROFILE

/* Define if we have the perf JIT-support library */
#cmakedefine01 LLVM_USE_PERF

/* Major version of the LLVM API */
#define LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR}

/* Minor version of the LLVM API */
#define LLVM_VERSION_MINOR ${LLVM_VERSION_MINOR}

/* Patch version of the LLVM API */
#define LLVM_VERSION_PATCH ${LLVM_VERSION_PATCH}

/* LLVM version string */
#define LLVM_VERSION_STRING "${PACKAGE_VERSION}"

/* Whether LLVM records statistics for use with GetStatistics(),
 * PrintStatistics() or PrintStatisticsJSON()
 */
#cmakedefine01 LLVM_FORCE_ENABLE_STATS

/* Define if we have z3 and want to build it */
#cmakedefine LLVM_WITH_Z3 ${LLVM_WITH_Z3}

/* Define if we have curl and want to use it */
#cmakedefine LLVM_ENABLE_CURL ${LLVM_ENABLE_CURL}

/* Define if we have cpp-httplib and want to use it */
#cmakedefine LLVM_ENABLE_HTTPLIB ${LLVM_ENABLE_HTTPLIB}

/* Define if zlib compression is available */
#cmakedefine01 LLVM_ENABLE_ZLIB

/* Define if zstd compression is available */
#cmakedefine01 LLVM_ENABLE_ZSTD

/* Define if LLVM was built with a dependency to the libtensorflow dynamic library */
#cmakedefine LLVM_HAVE_TF_API

/* Define to 1 if you have the <sysexits.h> header file. */
#cmakedefine HAVE_SYSEXITS_H ${HAVE_SYSEXITS_H}

/* Define if the xar_open() function is supported on this platform. */
#cmakedefine LLVM_HAVE_LIBXAR ${LLVM_HAVE_LIBXAR}

/* Define if building libLLVM shared library */
#cmakedefine LLVM_BUILD_LLVM_DYLIB

/* Define if building LLVM with BUILD_SHARED_LIBS */
#cmakedefine LLVM_BUILD_SHARED_LIBS

/* Define if building LLVM with LLVM_FORCE_USE_OLD_TOOLCHAIN_LIBS */
#cmakedefine LLVM_FORCE_USE_OLD_TOOLCHAIN ${LLVM_FORCE_USE_OLD_TOOLCHAIN}

/* Define if llvm_unreachable should be optimized with undefined behavior
 * in non assert builds */
#cmakedefine01 LLVM_UNREACHABLE_OPTIMIZE

/* Define to 1 if you have the DIA SDK installed, and to 0 if you don't. */
#cmakedefine01 LLVM_ENABLE_DIA_SDK

#endif
