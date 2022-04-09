/*===------- llvm/Config/llvm-config.h - llvm configuration -------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

/* This is a manual port of config.h.cmake for the symbols that do not change
   based on platform. Those that do change should not be defined here and
   instead use Bazel cc_library defines. Some attempt has been made to extract
   such symbols that do vary based on platform (for the platforms we care about)
   into Bazel defines, but it is by no means complete, so if you see something
   that looks wrong, it probably is. */


/* This file enumerates variables from the LLVM configuration so that they
   can be in exported headers and won't override package specific directives.
   This is a C header that can be included in the llvm-c headers. */

#ifndef LLVM_CONFIG_H
#define LLVM_CONFIG_H

/* Define if LLVM_ENABLE_DUMP is enabled */
/* #undef LLVM_ENABLE_DUMP */

/* Target triple LLVM will generate code for by default */
/* LLVM_DEFAULT_TARGET_TRIPLE defined in Bazel */

/* Define if threads enabled */
#define LLVM_ENABLE_THREADS 1

/* Has gcc/MSVC atomic intrinsics */
#define LLVM_HAS_ATOMICS 1

/* Host triple LLVM will be executed on */
/* LLVM_HOST_TRIPLE defined in Bazel */

/* LLVM architecture name for the native architecture, if available */
/* LLVM_NATIVE_ARCH defined in Bazel */

/* LLVM name for the native AsmParser init function, if available */
/* LLVM_NATIVE_ASMPARSER defined in Bazel */

/* LLVM name for the native AsmPrinter init function, if available */
/* LLVM_NATIVE_ASMPRINTER defined in Bazel */

/* LLVM name for the native Disassembler init function, if available */
/* LLVM_NATIVE_DISASSEMBLER defined in Bazel */

/* LLVM name for the native Target init function, if available */
/* LLVM_NATIVE_TARGET defined in Bazel */

/* LLVM name for the native TargetInfo init function, if available */
/* LLVM_NATIVE_TARGETINFO defined in Bazel */

/* LLVM name for the native target MC init function, if available */
/* LLVM_NATIVE_TARGETMC defined in Bazel */

/* LLVM name for the native target MCA init function, if available */
/* LLVM_NATIVE_TARGETMCA defined in Bazel */

/* Define if this is Unixish platform */
/* LLVM_ON_UNIX defined in Bazel */

/* Define if we have the Intel JIT API runtime support library */
#define LLVM_USE_INTEL_JITEVENTS 0

/* Define if we have the oprofile JIT-support library */
#define LLVM_USE_OPROFILE 0

/* Define if we have the perf JIT-support library */
#define LLVM_USE_PERF 0

/* Major version of the LLVM API */
#define LLVM_VERSION_MAJOR 14

/* Minor version of the LLVM API */
#define LLVM_VERSION_MINOR 0

/* Patch version of the LLVM API */
#define LLVM_VERSION_PATCH 0

/* LLVM version string */
#define LLVM_VERSION_STRING "14.0.0git"

/* Whether LLVM records statistics for use with GetStatistics(),
 * PrintStatistics() or PrintStatisticsJSON()
 */
#define LLVM_FORCE_ENABLE_STATS 0

/* Define if we have z3 and want to build it */
/* #undef LLVM_WITH_Z3 */

/* Define if we have curl and want to use it */
/* #undef LLVM_ENABLE_CURL */

/* Define if LLVM was built with a dependency to the libtensorflow dynamic library */
/* #undef LLVM_HAVE_TF_API */

/* Define if LLVM was built with a dependency to the tensorflow compiler */
/* #undef LLVM_HAVE_TF_AOT */

/* Define to 1 if you have the <sysexits.h> header file. */
/* HAVE_SYSEXITS_H defined in Bazel */

/* Define if the xar_open() function is supported this platform. */
/* #undef HAVE_LIBXAR */

/* Define if building libLLVM shared library */
/* #undef LLVM_BUILD_LLVM_DYLIB */

/* Define if building LLVM with BUILD_SHARED_LIBS */
/* #undef LLVM_BUILD_SHARED_LIBS */

/* Define if building LLVM with LLVM_FORCE_USE_OLD_TOOLCHAIN_LIBS */
/* #undef LLVM_FORCE_USE_OLD_TOOLCHAIN ${LLVM_FORCE_USE_OLD_TOOLCHAIN} */

/* Define if llvm_unreachable should be optimized with undefined behavior
 * in non assert builds */
#define LLVM_UNREACHABLE_OPTIMIZE 1

/* Define to 1 if you have the DIA SDK installed, and to 0 if you don't. */
#define LLVM_ENABLE_DIA_SDK 0

#endif
