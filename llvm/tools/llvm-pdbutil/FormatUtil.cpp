//===- FormatUtil.cpp ----------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FormatUtil.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::pdb;

std::string llvm::pdb::typesetItemList(ArrayRef<std::string> Opts,
                                       uint32_t IndentLevel, uint32_t GroupSize,
                                       StringRef Sep) {
  std::string Result;
  while (!Opts.empty()) {
    ArrayRef<std::string> ThisGroup;
    ThisGroup = Opts.take_front(GroupSize);
    Opts = Opts.drop_front(ThisGroup.size());
    Result += join(ThisGroup, Sep);
    if (!Opts.empty()) {
      Result += Sep;
      Result += "\n";
      Result += formatv("{0}", fmt_repeat(' ', IndentLevel));
    }
  }
  return Result;
}

std::string llvm::pdb::typesetStringList(uint32_t IndentLevel,
                                         ArrayRef<StringRef> Strings) {
  std::string Result = "[";
  for (const auto &S : Strings) {
    Result += formatv("\n{0}{1}", fmt_repeat(' ', IndentLevel), S);
  }
  Result += "]";
  return Result;
}

std::string llvm::pdb::formatSegmentOffset(uint16_t Segment, uint32_t Offset) {
  return formatv("{0:4}:{1:4}", Segment, Offset);
}
