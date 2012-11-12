//===- SimplifyLibCalls.h - Library call simplifier -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an interface to build some C language libcalls for
// optimization passes that need to call the various functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SIMPLIFYLIBCALLS_H
#define LLVM_TRANSFORMS_UTILS_SIMPLIFYLIBCALLS_H

namespace llvm {
  class Value;
  class CallInst;
  class DataLayout;
  class Instruction;
  class TargetLibraryInfo;
  class LibCallSimplifierImpl;

  /// LibCallSimplifier - This class implements a collection of optimizations
  /// that replace well formed calls to library functions with a more optimal
  /// form.  For example, replacing 'printf("Hello!")' with 'puts("Hello!")'.
  class LibCallSimplifier {
    /// Impl - A pointer to the actual implementation of the library call
    /// simplifier.
    LibCallSimplifierImpl *Impl;
  public:
    LibCallSimplifier(const DataLayout *TD, const TargetLibraryInfo *TLI);
    virtual ~LibCallSimplifier();

    /// optimizeCall - Take the given call instruction and return a more
    /// optimal value to replace the instruction with or 0 if a more
    /// optimal form can't be found.  Note that the returned value may
    /// be equal to the instruction being optimized.  In this case all
    /// other instructions that use the given instruction were modified
    /// and the given instruction is dead.
    Value *optimizeCall(CallInst *CI);

    /// replaceAllUsesWith - This method is used when the library call
    /// simplifier needs to replace instructions other than the library
    /// call being modified.
    virtual void replaceAllUsesWith(Instruction *I, Value *With) const;
  };
} // End llvm namespace

#endif
