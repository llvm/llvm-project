//===-- LVLine.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LVLine class, which is used to describe a debug
// information line.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVLINE_H
#define LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVLINE_H

#include "llvm/DebugInfo/LogicalView/Core/LVElement.h"

namespace llvm {
namespace logicalview {

enum class LVLineKind {
  IsBasicBlock,
  IsDiscriminator,
  IsEndSequence,
  IsEpilogueBegin,
  IsLineDebug,
  IsLineAssembler,
  IsNewStatement, // Shared with CodeView 'IsStatement' flag.
  IsPrologueEnd,
  IsAlwaysStepInto, // CodeView
  IsNeverStepInto,  // CodeView
  LastEntry
};
using LVLineKindSet = std::set<LVLineKind>;

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVLINE_H
