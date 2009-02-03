//===- AlphaJITInfo.h - Alpha impl. of the JIT interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Alpha implementation of the TargetJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ALPHA_JITINFO_H
#define ALPHA_JITINFO_H

#include "llvm/Target/TargetJITInfo.h"

namespace llvm {
  class TargetMachine;

  class AlphaJITInfo : public TargetJITInfo {
  protected:
    TargetMachine &TM;
  public:
    explicit AlphaJITInfo(TargetMachine &tm) : TM(tm)
    { useGOT = true; }

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
  private:
    static const unsigned GOToffset = 4096;

  };
}

#endif
