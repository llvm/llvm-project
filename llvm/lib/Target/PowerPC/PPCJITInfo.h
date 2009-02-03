//===- PPCJITInfo.h - PowerPC impl. of the JIT interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_JITINFO_H
#define POWERPC_JITINFO_H

#include "llvm/Target/TargetJITInfo.h"

namespace llvm {
  class PPCTargetMachine;

  class PPCJITInfo : public TargetJITInfo {
  protected:
    PPCTargetMachine &TM;
    bool is64Bit;
  public:
    PPCJITInfo(PPCTargetMachine &tm, bool tmIs64Bit) : TM(tm) {
      useGOT = 0;
      is64Bit = tmIs64Bit;
    }

    virtual void *emitFunctionStub(const Function* F, void *Fn,
                                   MachineCodeEmitter &MCE);
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn);
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase);
    
    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New);
  };
}

#endif
