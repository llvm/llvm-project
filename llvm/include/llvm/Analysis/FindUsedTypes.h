//===- llvm/Analysis/FindUsedTypes.h - Find all Types in use ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass is used to seek out all of the types in use by the program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_FINDUSEDTYPES_H
#define LLVM_ANALYSIS_FINDUSEDTYPES_H

#include "llvm/Pass.h"
#include <set>
class SymbolTable;
class Type;

class FindUsedTypes : public Pass {
  std::set<const Type *> UsedTypes;
public:
  /// getTypes - After the pass has been run, return the set containing all of
  /// the types used in the module.
  ///
  const std::set<const Type *> &getTypes() const { return UsedTypes; }

  /// Print the types found in the module.  If the optional Module parameter is
  /// passed in, then the types are printed symbolically if possible, using the
  /// symbol table from the module.
  ///
  void print(std::ostream &o, const Module *M) const;

private:
  /// IncorporateType - Incorporate one type and all of its subtypes into the
  /// collection of used types.
  ///
  void IncorporateType(const Type *Ty);

  /// IncorporateSymbolTable - Include any named types.
  ///
  void IncorporateSymbolTable(const SymbolTable &ST);

public:
  /// run - This incorporates all types used by the specified module
  bool run(Module &M);

  /// getAnalysisUsage - We do not modify anything.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  // stub - dummy function, just ignore it
  static void stub();
};

// Make sure that any clients of this file link in PostDominators.cpp
static IncludeFile
FIND_USED_TYPES_INCLUDE_FILE((void*)&FindUsedTypes::stub);

#endif
