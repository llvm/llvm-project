//===- CoroInternal.h - Internal Coroutine interfaces ---------*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Common definitions/declarations used internally by coroutine lowering passes.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_COROUTINES_COROINTERNAL_H
#define LLVM_LIB_TRANSFORMS_COROUTINES_COROINTERNAL_H

#include "llvm/Transforms/Coroutines.h"

namespace llvm {

class PassRegistry;

void initializeCoroEarlyPass(PassRegistry &);
void initializeCoroSplitPass(PassRegistry &);
void initializeCoroElidePass(PassRegistry &);
void initializeCoroCleanupPass(PassRegistry &);

}

#endif
