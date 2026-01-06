//===-- LVSourceLanguage.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LVSourceLanguage struct, a unified representation of
// the source language used in a compile unit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSOURCELANGUAGE_H
#define LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSOURCELANGUAGE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace logicalview {

/// A source language supported by any of the debug info representations.
struct LVSourceLanguage {
  static constexpr unsigned TagDwarf = 0x00;
  static constexpr unsigned TagCodeView = 0x01;

  enum TaggedLanguage : uint32_t {
    Invalid = -1U,

  // DWARF
#define HANDLE_DW_LANG(ID, NAME, LOWER_BOUND, VERSION, VENDOR)                 \
  DW_LANG_##NAME = (TagDwarf << 16) | ID,
#include "llvm/BinaryFormat/Dwarf.def"
  // CodeView
#define CV_LANGUAGE(NAME, ID) CV_LANG_##NAME = (TagCodeView << 16) | ID,
#include "llvm/DebugInfo/CodeView/CodeViewLanguages.def"
  };

  LVSourceLanguage() = default;
  LVSourceLanguage(llvm::dwarf::SourceLanguage SL)
      : LVSourceLanguage(TagDwarf, SL) {}
  LVSourceLanguage(llvm::codeview::SourceLanguage SL)
      : LVSourceLanguage(TagCodeView, SL) {}
  bool operator==(const LVSourceLanguage &SL) const {
    return get() == SL.get();
  }
  bool operator==(const LVSourceLanguage::TaggedLanguage &TL) const {
    return get() == TL;
  }

  bool isValid() const { return Language != Invalid; }
  TaggedLanguage get() const { return Language; }
  LLVM_ABI StringRef getName() const;

private:
  TaggedLanguage Language = Invalid;

  LVSourceLanguage(unsigned Tag, unsigned Lang)
      : Language(static_cast<TaggedLanguage>((Tag << 16) | Lang)) {}
  unsigned getTag() const { return Language >> 16; }
  unsigned getLang() const { return Language & 0xffff; }
};

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSOURCELANGUAGE_H
