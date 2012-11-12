//===- llvm/Analysis/FindUsedTypes.h - Find all Types in use ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is used to seek out all of the types in use by the program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_FINDUSEDTYPES_H
#define LLVM_ANALYSIS_FINDUSEDTYPES_H

#include "llvm/ADT/SetVector.h"
#include "llvm/Pass.h"

namespace llvm {

class Type;
class Value;

class FindUsedTypes : public ModulePass {
  SetVector<Type *> UsedTypes;
public:
  static char ID; // Pass identification, replacement for typeid
  FindUsedTypes() : ModulePass(ID) {
    initializeFindUsedTypesPass(*PassRegistry::getPassRegistry());
  }

  /// getTypes - After the pass has been run, return the set containing all of
  /// the types used in the module.
  ///
  const SetVector<Type *> &getTypes() const { return UsedTypes; }

  /// Print the types found in the module.  If the optional Module parameter is
  /// passed in, then the types are printed symbolically if possible, using the
  /// symbol table from the module.
  ///
  void print(raw_ostream &o, const Module *M) const;

private:
  /// IncorporateType - Incorporate one type and all of its subtypes into the
  /// collection of used types.
  ///
  void IncorporateType(Type *Ty);

  /// IncorporateValue - Incorporate all of the types used by this value.
  ///
  void IncorporateValue(const Value *V);

public:
  /// run - This incorporates all types used by the specified module
  bool runOnModule(Module &M);

  /// getAnalysisUsage - We do not modify anything.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

} // End llvm namespace

#endif
