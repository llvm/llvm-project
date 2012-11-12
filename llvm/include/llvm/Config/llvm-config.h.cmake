/*===-- llvm/config/llvm-config.h - llvm configure variable -------*- C -*-===*/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is distributed under the University of Illinois Open Source      */
/* License. See LICENSE.TXT for details.                                      */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

/* This file enumerates all of the llvm variables from configure so that
   they can be in exported headers and won't override package specific
   directives.  This is a C file so we can include it in the llvm-c headers.  */

/* To avoid multiple inclusions of these variables when we include the exported
   headers and config.h, conditionally include these.  */
/* TODO: This is a bit of a hack.  */
#ifndef CONFIG_H

/* Installation directory for binary executables */
#cmakedefine LLVM_BINDIR "${LLVM_BINDIR}"

/* Time at which LLVM was configured */
#cmakedefine LLVM_CONFIGTIME "${LLVM_CONFIGTIME}"

/* Installation directory for data files */
#cmakedefine LLVM_DATADIR "${LLVM_DATADIR}"

/* Target triple LLVM will generate code for by default */
#cmakedefine LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}"

/* Installation directory for documentation */
#cmakedefine LLVM_DOCSDIR "${LLVM_DOCSDIR}"

/* Define if threads enabled */
#cmakedefine01 LLVM_ENABLE_THREADS

/* Installation directory for config files */
#cmakedefine LLVM_ETCDIR "${LLVM_ETCDIR}"

/* Has gcc/MSVC atomic intrinsics */
#cmakedefine01 LLVM_HAS_ATOMICS

/* Host triple LLVM will be executed on */
#cmakedefine LLVM_HOSTTRIPLE "${LLVM_HOSTTRIPLE}"

/* Installation directory for include files */
#cmakedefine LLVM_INCLUDEDIR "${LLVM_INCLUDEDIR}"

/* Installation directory for .info files */
#cmakedefine LLVM_INFODIR "${LLVM_INFODIR}"

/* Installation directory for libraries */
#cmakedefine LLVM_LIBDIR "${LLVM_LIBDIR}"

/* Installation directory for man pages */
#cmakedefine LLVM_MANDIR "${LLVM_MANDIR}"

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

/* Define if this is Unixish platform */
#cmakedefine LLVM_ON_UNIX ${LLVM_ON_UNIX}

/* Define if this is Win32ish platform */
#cmakedefine LLVM_ON_WIN32 ${LLVM_ON_WIN32}

/* Define to path to circo program if found or 'echo circo' otherwise */
#cmakedefine LLVM_PATH_CIRCO "${LLVM_PATH_CIRCO}"

/* Define to path to dot program if found or 'echo dot' otherwise */
#cmakedefine LLVM_PATH_DOT "${LLVM_PATH_DOT}"

/* Define to path to dotty program if found or 'echo dotty' otherwise */
#cmakedefine LLVM_PATH_DOTTY "${LLVM_PATH_DOTTY}"

/* Define to path to fdp program if found or 'echo fdp' otherwise */
#cmakedefine LLVM_PATH_FDP "${LLVM_PATH_FDP}"

/* Define to path to Graphviz program if found or 'echo Graphviz' otherwise */
#cmakedefine LLVM_PATH_GRAPHVIZ "${LLVM_PATH_GRAPHVIZ}"

/* Define to path to gv program if found or 'echo gv' otherwise */
#cmakedefine LLVM_PATH_GV "${LLVM_PATH_GV}"

/* Define to path to neato program if found or 'echo neato' otherwise */
#cmakedefine LLVM_PATH_NEATO "${LLVM_PATH_NEATO}"

/* Define to path to twopi program if found or 'echo twopi' otherwise */
#cmakedefine LLVM_PATH_TWOPI "${LLVM_PATH_TWOPI}"

/* Define to path to xdot.py program if found or 'echo xdot.py' otherwise */
#cmakedefine LLVM_PATH_XDOT_PY "${LLVM_PATH_XDOT_PY}"

/* Installation prefix directory */
#cmakedefine LLVM_PREFIX "${LLVM_PREFIX}"

/* Major version of the LLVM API */
#cmakedefine LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR}

/* Minor version of the LLVM API */
#cmakedefine LLVM_VERSION_MINOR ${LLVM_VERSION_MINOR}

#endif
