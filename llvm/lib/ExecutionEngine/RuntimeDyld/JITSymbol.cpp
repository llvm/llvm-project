//===----------- JITSymbol.cpp - JITSymbol class implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// JITSymbol class implementation plus helper functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Object/SymbolicFile.h"

using namespace llvm;

JITSymbolFlags llvm::JITSymbolFlags::fromGlobalValue(const GlobalValue &GV) {
  JITSymbolFlags Flags = JITSymbolFlags::None;
  if (GV.hasWeakLinkage() || GV.hasLinkOnceLinkage())
    Flags |= JITSymbolFlags::Weak;
  if (GV.hasCommonLinkage())
    Flags |= JITSymbolFlags::Common;
  if (!GV.hasLocalLinkage() && !GV.hasHiddenVisibility())
    Flags |= JITSymbolFlags::Exported;
  return Flags;
}

JITSymbolFlags
llvm::JITSymbolFlags::fromObjectSymbol(const object::BasicSymbolRef &Symbol) {
  JITSymbolFlags Flags = JITSymbolFlags::None;
  if (Symbol.getFlags() & object::BasicSymbolRef::SF_Weak)
    Flags |= JITSymbolFlags::Weak;
  if (Symbol.getFlags() & object::BasicSymbolRef::SF_Common)
    Flags |= JITSymbolFlags::Common;
  if (Symbol.getFlags() & object::BasicSymbolRef::SF_Exported)
    Flags |= JITSymbolFlags::Exported;
  return Flags;
}
