//===-- LVType.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LVType class, which is used to describe a debug
// information type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVTYPE_H
#define LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVTYPE_H

#include "llvm/DebugInfo/LogicalView/Core/LVElement.h"

namespace llvm {
namespace logicalview {

enum class LVTypeKind {
  IsBase,
  IsConst,
  IsEnumerator,
  IsImport,
  IsImportDeclaration,
  IsImportModule,
  IsPointer,
  IsPointerMember,
  IsReference,
  IsRestrict,
  IsRvalueReference,
  IsSubrange,
  IsTemplateParam,
  IsTemplateTemplateParam,
  IsTemplateTypeParam,
  IsTemplateValueParam,
  IsTypedef,
  IsUnaligned,
  IsUnspecified,
  IsVolatile,
  IsModifier, // CodeView - LF_MODIFIER
  LastEntry
};
using LVTypeKindSelection = std::set<LVTypeKind>;

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVTYPE_H
