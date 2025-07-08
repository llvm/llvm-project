//===-- LVSourceLanguage.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements LVSourceLanguage.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVSourceLanguage.h"
#include "llvm/DebugInfo/CodeView/EnumTables.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace llvm::logicalview;

StringRef LVSourceLanguage::getName() const {
  if (!isValid())
    return {};
  switch (getTag()) {
  case LVSourceLanguage::TagDwarf:
    return llvm::dwarf::LanguageString(getLang());
  case LVSourceLanguage::TagCodeView: {
    static auto LangNames = llvm::codeview::getSourceLanguageNames();
    return LangNames[getLang()].Name;
  }
  default:
    llvm_unreachable("Unsupported language");
  }
}
