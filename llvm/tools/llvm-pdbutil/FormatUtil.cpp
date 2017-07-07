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

std::string llvm::pdb::truncateStringBack(StringRef S, uint32_t MaxLen) {
  if (MaxLen == 0 || S.size() <= MaxLen || S.size() <= 3)
    return S;

  assert(MaxLen >= 3);
  uint32_t FinalLen = std::min<size_t>(S.size(), MaxLen - 3);
  S = S.take_front(FinalLen);
  return std::string(S) + std::string("...");
}

std::string llvm::pdb::truncateStringMiddle(StringRef S, uint32_t MaxLen) {
  if (MaxLen == 0 || S.size() <= MaxLen || S.size() <= 3)
    return S;

  assert(MaxLen >= 3);
  uint32_t FinalLen = std::min<size_t>(S.size(), MaxLen - 3);
  StringRef Front = S.take_front(FinalLen / 2);
  StringRef Back = S.take_back(Front.size());
  return std::string(Front) + std::string("...") + std::string(Back);
}

std::string llvm::pdb::truncateStringFront(StringRef S, uint32_t MaxLen) {
  if (MaxLen == 0 || S.size() <= MaxLen || S.size() <= 3)
    return S;

  assert(MaxLen >= 3);
  S = S.take_back(MaxLen - 3);
  return std::string("...") + std::string(S);
}

std::string llvm::pdb::truncateQuotedNameFront(StringRef Label, StringRef Name,
                                               uint32_t MaxLen) {
  uint32_t RequiredExtraChars = Label.size() + 1 + 2;
  if (MaxLen == 0 || RequiredExtraChars + Name.size() <= MaxLen)
    return formatv("{0} \"{1}\"", Label, Name).str();

  assert(MaxLen >= RequiredExtraChars);
  std::string TN = truncateStringFront(Name, MaxLen - RequiredExtraChars);
  return formatv("{0} \"{1}\"", Label, TN).str();
}

std::string llvm::pdb::truncateQuotedNameBack(StringRef Label, StringRef Name,
                                              uint32_t MaxLen) {
  uint32_t RequiredExtraChars = Label.size() + 1 + 2;
  if (MaxLen == 0 || RequiredExtraChars + Name.size() <= MaxLen)
    return formatv("{0} \"{1}\"", Label, Name).str();

  assert(MaxLen >= RequiredExtraChars);
  std::string TN = truncateStringBack(Name, MaxLen - RequiredExtraChars);
  return formatv("{0} \"{1}\"", Label, TN).str();
}

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
