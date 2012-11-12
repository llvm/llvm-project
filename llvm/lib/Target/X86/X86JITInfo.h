//===-- X86JITInfo.h - X86 implementation of the JIT interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86JITINFO_H
#define X86JITINFO_H

#include "llvm/Function.h"
#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/Target/TargetJITInfo.h"

namespace llvm {
  class X86TargetMachine;
  class X86Subtarget;

  class X86JITInfo : public TargetJITInfo {
    X86TargetMachine &TM;
    const X86Subtarget *Subtarget;
    uintptr_t PICBase;
    char* TLSOffset;
  public:
    explicit X86JITInfo(X86TargetMachine &tm);

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New);

    /// emitGlobalValueIndirectSym - Use the specified JITCodeEmitter object
    /// to emit an indirect symbol which contains the address of the specified
    /// ptr.
    virtual void *emitGlobalValueIndirectSym(const GlobalValue* GV, void *ptr,
                                             JITCodeEmitter &JCE);

    // getStubLayout - Returns the size and alignment of the largest call stub
    // on X86.
    virtual StubLayout getStubLayout();

    /// emitFunctionStub - Use the specified JITCodeEmitter object to emit a
    /// small native function that simply calls the function at the specified
    /// address.
    virtual void *emitFunctionStub(const Function* F, void *Target,
                                   JITCodeEmitter &JCE);

    /// getPICJumpTableEntry - Returns the value of the jumptable entry for the
    /// specific basic block.
    virtual uintptr_t getPICJumpTableEntry(uintptr_t BB, uintptr_t JTBase);

    /// getLazyResolverFunction - Expose the lazy resolver to the JIT.
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn);

    /// relocate - Before the JIT can run a block of code that has been emitted,
    /// it must rewrite the code to contain the actual addresses of any
    /// referenced global symbols.
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase);

    /// allocateThreadLocalMemory - Each target has its own way of
    /// handling thread local variables. This method returns a value only
    /// meaningful to the target.
    virtual char* allocateThreadLocalMemory(size_t size);

    /// setPICBase / getPICBase - Getter / setter of PICBase, used to compute
    /// PIC jumptable entry.
    void setPICBase(uintptr_t Base) { PICBase = Base; }
    uintptr_t getPICBase() const { return PICBase; }
  };
}

#endif
