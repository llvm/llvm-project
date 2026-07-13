//===---- CoreContainers.h - Symbol Containers for Core APIs ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Symbol container types for core ORC APIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_CORECONTAINERS_H
#define LLVM_EXECUTIONENGINE_ORC_CORECONTAINERS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"

#include <vector>

namespace llvm::orc {

class JITDylib;

/// A set of symbol names (represented by SymbolStringPtrs for
//         efficiency).
using SymbolNameSet = DenseSet<SymbolStringPtr>;

/// A vector of symbol names.
using SymbolNameVector = std::vector<SymbolStringPtr>;

/// A map from symbol names (as SymbolStringPtrs) to JITSymbols
/// (address/flags pairs).
using SymbolMap = DenseMap<SymbolStringPtr, ExecutorSymbolDef>;

/// A map from symbol names (as SymbolStringPtrs) to JITSymbolFlags.
using SymbolFlagsMap = DenseMap<SymbolStringPtr, JITSymbolFlags>;

/// A map from JITDylibs to sets of symbols.
using SymbolDependenceMap = DenseMap<JITDylib *, SymbolNameSet>;

} // End namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_CORECONTAINERS_H
