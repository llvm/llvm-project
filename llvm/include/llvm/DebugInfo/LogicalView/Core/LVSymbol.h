//===-- LVSymbol.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LVSymbol class, which is used to describe a debug
// information symbol.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSYMBOL_H
#define LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSYMBOL_H

#include "llvm/DebugInfo/LogicalView/Core/LVElement.h"

namespace llvm {
namespace logicalview {

enum class LVSymbolKind {
  IsCallSiteParameter,
  IsConstant,
  IsInheritance,
  IsMember,
  IsParameter,
  IsUnspecified,
  IsVariable,
  LastEntry
};
using LVSymbolKindSet = std::set<LVSymbolKind>;

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSYMBOL_H
