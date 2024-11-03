/*===------- clang/Config/config.h - llvm configuration -----------*- C -*-===*/
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

#ifdef CLANG_CONFIG_H
#error config.h can only be included once
#else
#define CLANG_CONFIG_H

/* Bug report URL. */
#define BUG_REPORT_URL "https://github.com/llvm/llvm-project/issues/"

/* Default to -fPIE and -pie on Linux. */
#define CLANG_DEFAULT_PIE_ON_LINUX 1

/* Default linker to use. */
#define CLANG_DEFAULT_LINKER ""

/* Default C++ stdlib to use. */
#define CLANG_DEFAULT_CXX_STDLIB ""

/* Default runtime library to use. */
#define CLANG_DEFAULT_RTLIB ""

/* Default unwind library to use. */
#define CLANG_DEFAULT_UNWINDLIB ""

/* Default objcopy to use */
#define CLANG_DEFAULT_OBJCOPY "objcopy"

/* Default OpenMP runtime used by -fopenmp. */
#define CLANG_DEFAULT_OPENMP_RUNTIME "libomp"

/* Default architecture for SystemZ. */
#define CLANG_SYSTEMZ_DEFAULT_ARCH "z10"

/* Multilib basename for libdir. */
#define CLANG_INSTALL_LIBDIR_BASENAME "lib"

/* Relative directory for resource files */
#define CLANG_RESOURCE_DIR ""

/* Directories clang will search for headers */
#define C_INCLUDE_DIRS ""

/* Directories clang will search for configuration files */
/* #undef CLANG_CONFIG_FILE_SYSTEM_DIR */
/* #undef CLANG_CONFIG_FILE_USER_DIR */

/* Default <path> to all compiler invocations for --sysroot=<path>. */
#define DEFAULT_SYSROOT ""

/* Directory where gcc is installed. */
#define GCC_INSTALL_PREFIX ""

/* Define if we have libxml2 */
/* #undef CLANG_HAVE_LIBXML */

/* Define if we have sys/resource.h (rlimits) */
/* CLANG_HAVE_RLIMITS defined conditionally below */

/* Linker version detected at compile time. */
/* #undef HOST_LINK_VERSION */

/* pass --build-id to ld */
/* #undef ENABLE_LINKER_BUILD_ID */

/* enable x86 relax relocations by default */
#define ENABLE_X86_RELAX_RELOCATIONS 1

/* enable IEEE binary128 as default long double format on PowerPC Linux. */
#define PPC_LINUX_DEFAULT_IEEELONGDOUBLE 0

/* Enable the experimental new pass manager by default */
#define ENABLE_EXPERIMENTAL_NEW_PASS_MANAGER 0

/* Enable each functionality of modules */
#define CLANG_ENABLE_ARCMT 1
#define CLANG_ENABLE_OBJC_REWRITER 1
#define CLANG_ENABLE_STATIC_ANALYZER 1

/* Spawn a new process clang.exe for the CC1 tool invocation, when necessary */
#define CLANG_SPAWN_CC1 0

/* Whether to enable opaque pointers by default */
#define CLANG_ENABLE_OPAQUE_POINTERS_INTERNAL 1

/* Directly provide definitions here behind platform preprocessor definitions.
 * The preprocessor conditions are sufficient to handle all of the configuration
 * on platforms targeted by Bazel, and defining these here more faithfully
 * matches how the users of this header expect things to work with CMake.
 */

#ifndef _WIN32
#define CLANG_HAVE_RLIMITS 1
#endif

#endif
