//===- InlineInfo.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/FileEntry.h"
#include "llvm/DebugInfo/GSYM/InlineInfo.h"
#include <algorithm>
#include <inttypes.h>

using namespace llvm;
using namespace gsym;


raw_ostream &llvm::gsym::operator<<(raw_ostream &OS, const InlineInfo &II) {
  if (!II.isValid())
    return OS;
  bool First = true;
  for (auto Range : II.Ranges) {
    if (First)
      First = false;
    else
      OS << ' ';
    OS << Range;
  }
  OS << " Name = " << HEX32(II.Name) << ", CallFile = " << II.CallFile
     << ", CallLine = " << II.CallFile << '\n';
  for (const auto &Child : II.Children)
    OS << Child;
  return OS;
}

static bool getInlineStackHelper(const InlineInfo &II, uint64_t Addr,
    std::vector<const InlineInfo *> &InlineStack) {
  if (II.Ranges.contains(Addr)) {
    // If this is the top level that represents the concrete function,
    // there will be no name and we shoud clear the inline stack. Otherwise
    // we have found an inline call stack that we need to insert.
    if (II.Name != 0)
      InlineStack.insert(InlineStack.begin(), &II);
    for (const auto &Child : II.Children) {
      if (::getInlineStackHelper(Child, Addr, InlineStack))
        break;
    }
    return !InlineStack.empty();
  }
  return false;
}

llvm::Optional<InlineInfo::InlineArray> InlineInfo::getInlineStack(uint64_t Addr) const {
  InlineArray Result;
  if (getInlineStackHelper(*this, Addr, Result))
    return Result;
  return llvm::None;
}
