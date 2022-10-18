//===-- LVScope.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LVScope class, which is used to describe a debug
// information scope.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSCOPE_H
#define LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSCOPE_H

#include "llvm/DebugInfo/LogicalView/Core/LVElement.h"
#include "llvm/DebugInfo/LogicalView/Core/LVSort.h"
#include <set>

namespace llvm {
namespace logicalview {

enum class LVScopeKind {
  IsAggregate,
  IsArray,
  IsBlock,
  IsCallSite,
  IsCatchBlock,
  IsClass,
  IsCompileUnit,
  IsEntryPoint,
  IsEnumeration,
  IsFunction,
  IsFunctionType,
  IsInlinedFunction,
  IsLabel,
  IsLexicalBlock,
  IsMember,
  IsNamespace,
  IsRoot,
  IsStructure,
  IsSubprogram,
  IsTemplate,
  IsTemplateAlias,
  IsTemplatePack,
  IsTryBlock,
  IsUnion,
  LastEntry
};
using LVScopeKindSet = std::set<LVScopeKind>;

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSCOPE_H
