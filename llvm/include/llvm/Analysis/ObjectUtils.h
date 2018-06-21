//===- Analysis/ObjectUtils.h - analysis utils for object files -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_OBJECT_UTILS_H
#define LLVM_ANALYSIS_OBJECT_UTILS_H

#include "llvm/IR/GlobalVariable.h"

namespace llvm {

/// True if GV can be left out of the object symbol table. This is the case
/// for linkonce_odr values whose address is not significant. While legal, it is
/// not normally profitable to omit them from the .o symbol table. Using this
/// analysis makes sense when the information can be passed down to the linker
/// or we are in LTO.
inline bool canBeOmittedFromSymbolTable(const GlobalValue *GV) {
  if (!GV->hasLinkOnceODRLinkage())
    return false;

  // We assume that anyone who sets global unnamed_addr on a non-constant knows
  // what they're doing.
  if (GV->hasGlobalUnnamedAddr())
    return true;

  // If it is a non constant variable, it needs to be uniqued across shared
  // objects.
  if (auto *Var = dyn_cast<GlobalVariable>(GV))
    if (!Var->isConstant())
      return false;

  return GV->hasAtLeastLocalUnnamedAddr();
}

}

#endif
