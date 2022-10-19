//===-- LVObject.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LVObject class, which is used to describe a debug
// information object.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVOBJECT_H
#define LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVOBJECT_H

#include "llvm/DebugInfo/LogicalView/Core/LVSupport.h"
#include <string>

namespace llvm {
namespace logicalview {

using LVHalf = uint16_t;
using LVOffset = uint64_t;

class LVObject {
  LVOffset Offset = 0;

protected:
  // Get a string representation for the given number and discriminator.
  std::string lineAsString(uint32_t LineNumber, LVHalf Discriminator,
                           bool ShowZero) const;

  // Get a string representation for the given number.
  std::string referenceAsString(uint32_t LineNumber, bool Spaces) const;

  // Print the Filename or Pathname.
  // Empty implementation for those objects that do not have any user
  // source file references, such as debug locations.
  virtual void printFileIndex(raw_ostream &OS, bool Full = true) const {}

public:
  LVObject() = default;
  virtual ~LVObject() = default;

  // True if the scope has been named.
  virtual bool isNamed() const { return false; }

  // DIE offset.
  LVOffset getOffset() const { return Offset; }

  virtual StringRef getName() const { return StringRef(); }

  std::string lineNumberAsStringStripped(bool ShowZero = false) const;
};

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVOBJECT_H
