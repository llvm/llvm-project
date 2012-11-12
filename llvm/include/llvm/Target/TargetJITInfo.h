//===- Target/TargetJITInfo.h - Target Information for JIT ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an abstract interface used by the Just-In-Time code
// generator to perform target-specific activities, such as emitting stubs.  If
// a TargetMachine supports JIT code generation, it should provide one of these
// objects through the getJITInfo() method.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETJITINFO_H
#define LLVM_TARGET_TARGETJITINFO_H

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {
  class Function;
  class GlobalValue;
  class JITCodeEmitter;
  class MachineRelocation;

  /// TargetJITInfo - Target specific information required by the Just-In-Time
  /// code generator.
  class TargetJITInfo {
    virtual void anchor();
  public:
    virtual ~TargetJITInfo() {}

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New) = 0;

    /// emitGlobalValueIndirectSym - Use the specified JITCodeEmitter object
    /// to emit an indirect symbol which contains the address of the specified
    /// ptr.
    virtual void *emitGlobalValueIndirectSym(const GlobalValue* GV, void *ptr,
                                             JITCodeEmitter &JCE) {
      llvm_unreachable("This target doesn't implement "
                       "emitGlobalValueIndirectSym!");
    }

    /// Records the required size and alignment for a call stub in bytes.
    struct StubLayout {
      size_t Size;
      size_t Alignment;
    };
    /// Returns the maximum size and alignment for a call stub on this target.
    virtual StubLayout getStubLayout() {
      llvm_unreachable("This target doesn't implement getStubLayout!");
    }

    /// emitFunctionStub - Use the specified JITCodeEmitter object to emit a
    /// small native function that simply calls the function at the specified
    /// address.  The JITCodeEmitter must already have storage allocated for the
    /// stub.  Return the address of the resultant function, which may have been
    /// aligned from the address the JCE was set up to emit at.
    virtual void *emitFunctionStub(const Function* F, void *Target,
                                   JITCodeEmitter &JCE) {
      llvm_unreachable("This target doesn't implement emitFunctionStub!");
    }

    /// getPICJumpTableEntry - Returns the value of the jumptable entry for the
    /// specific basic block.
    virtual uintptr_t getPICJumpTableEntry(uintptr_t BB, uintptr_t JTBase) {
      llvm_unreachable("This target doesn't implement getPICJumpTableEntry!");
    }

    /// LazyResolverFn - This typedef is used to represent the function that
    /// unresolved call points should invoke.  This is a target specific
    /// function that knows how to walk the stack and find out which stub the
    /// call is coming from.
    typedef void (*LazyResolverFn)();

    /// JITCompilerFn - This typedef is used to represent the JIT function that
    /// lazily compiles the function corresponding to a stub.  The JIT keeps
    /// track of the mapping between stubs and LLVM Functions, the target
    /// provides the ability to figure out the address of a stub that is called
    /// by the LazyResolverFn.
    typedef void* (*JITCompilerFn)(void *);

    /// getLazyResolverFunction - This method is used to initialize the JIT,
    /// giving the target the function that should be used to compile a
    /// function, and giving the JIT the target function used to do the lazy
    /// resolving.
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn) {
      llvm_unreachable("Not implemented for this target!");
    }

    /// relocate - Before the JIT can run a block of code that has been emitted,
    /// it must rewrite the code to contain the actual addresses of any
    /// referenced global symbols.
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase) {
      assert(NumRelocs == 0 && "This target does not have relocations!");
    }
    

    /// allocateThreadLocalMemory - Each target has its own way of
    /// handling thread local variables. This method returns a value only
    /// meaningful to the target.
    virtual char* allocateThreadLocalMemory(size_t size) {
      llvm_unreachable("This target does not implement thread local storage!");
    }

    /// needsGOT - Allows a target to specify that it would like the
    /// JIT to manage a GOT for it.
    bool needsGOT() const { return useGOT; }

    /// hasCustomConstantPool - Allows a target to specify that constant
    /// pool address resolution is handled by the target.
    virtual bool hasCustomConstantPool() const { return false; }

    /// hasCustomJumpTables - Allows a target to specify that jumptables
    /// are emitted by the target.
    virtual bool hasCustomJumpTables() const { return false; }

    /// allocateSeparateGVMemory - If true, globals should be placed in
    /// separately allocated heap memory rather than in the same
    /// code memory allocated by JITCodeEmitter.
    virtual bool allocateSeparateGVMemory() const { return false; }
  protected:
    bool useGOT;
  };
} // End llvm namespace

#endif
