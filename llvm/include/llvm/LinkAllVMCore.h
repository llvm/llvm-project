//===- LinkAllVMCore.h - Reference All VMCore Code --------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all the object modules of the VMCore library so
// that tools like llc, opt, and lli can ensure they are linked with all symbols
// from libVMCore.a It should only be used from a tool's main program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKALLVMCORE_H
#define LLVM_LINKALLVMCORE_H

#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/InlineAsm.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TimeValue.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/MathExtras.h"
#include <cstdlib>

namespace {
  struct ForceVMCoreLinking {
    ForceVMCoreLinking() {
      // We must reference VMCore in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;
      (void)new llvm::Module("", llvm::getGlobalContext());
      (void)new llvm::UnreachableInst(llvm::getGlobalContext());
      (void)    llvm::createVerifierPass(); 
    }
  } ForceVMCoreLinking;
}

#endif
