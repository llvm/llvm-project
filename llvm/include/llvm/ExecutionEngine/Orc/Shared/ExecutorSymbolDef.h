//===--------- ExecutorSymbolDef.h - (Addr, Flags) pair ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represents a defining location for a JIT symbol.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_EXECUTORSYMBOLDEF_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_EXECUTORSYMBOLDEF_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"

namespace llvm {
namespace orc {

/// Represents a defining location for a JIT symbol.
class ExecutorSymbolDef {
public:
  ExecutorSymbolDef() = default;
  ExecutorSymbolDef(ExecutorAddr Addr, JITSymbolFlags Flags)
    : Addr(Addr), Flags(Flags) {}

  const ExecutorAddr &getAddress() const { return Addr; }

  const JITSymbolFlags &getFlags() const { return Flags; }

  void setFlags(JITSymbolFlags Flags) { this->Flags = Flags; }

  friend bool operator==(const ExecutorSymbolDef &LHS,
                         const ExecutorSymbolDef &RHS) {
    return LHS.getAddress() == RHS.getAddress() &&
           LHS.getFlags() == RHS.getFlags();
  }

  friend bool operator!=(const ExecutorSymbolDef &LHS,
                         const ExecutorSymbolDef &RHS) {
    return !(LHS == RHS);
  }

private:
  ExecutorAddr Addr;
  JITSymbolFlags Flags;
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_EXECUTORSYMBOLDEF_H
