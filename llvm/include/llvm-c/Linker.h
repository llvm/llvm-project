/*===-- llvm-c/Linker.h - Module Linker C Interface -------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file defines the C interface to the module/file/archive linker.       *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_LINKER_H
#define LLVM_C_LINKER_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
  LLVMLinkerDestroySource = 0, /* Allow source module to be destroyed. */
  LLVMLinkerPreserveSource = 1 /* Preserve the source module. */
} LLVMLinkerMode;


/* Links the source module into the destination module, taking ownership
 * of the source module away from the caller. Optionally returns a
 * human-readable description of any errors that occurred in linking.
 * OutMessage must be disposed with LLVMDisposeMessage. The return value
 * is true if an error occurred, false otherwise. */
LLVMBool LLVMLinkModules(LLVMModuleRef Dest, LLVMModuleRef Src,
                         LLVMLinkerMode Mode, char **OutMessage);

#ifdef __cplusplus
}
#endif

#endif
