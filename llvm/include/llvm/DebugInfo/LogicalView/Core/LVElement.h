//===-- LVElement.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LVElement class, which is used to describe a debug
// information element.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVELEMENT_H
#define LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVELEMENT_H

#include "llvm/DebugInfo/LogicalView/Core/LVObject.h"
#include "llvm/DebugInfo/LogicalView/Core/LVStringPool.h"
#include <set>

namespace llvm {
namespace logicalview {

enum class LVElementKind { Discarded, Global, Optimized, LastEntry };
using LVElementKindSet = std::set<LVElementKind>;

class LVElement : public LVObject {
  // Indexes in the String Pool.
  size_t NameIndex = 0;
  size_t FilenameIndex = 0;

public:
  LVElement() = default;
  virtual ~LVElement() = default;

  bool isNamed() const override { return NameIndex != 0; }

  StringRef getName() const override {
    return getStringPool().getString(NameIndex);
  }

  // Get pathname associated with the Element.
  StringRef getPathname() const {
    return getStringPool().getString(getFilenameIndex());
  }

  // Element type name.
  StringRef getTypeName() const;
  size_t getFilenameIndex() const { return FilenameIndex; }
};

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVELEMENT_H
