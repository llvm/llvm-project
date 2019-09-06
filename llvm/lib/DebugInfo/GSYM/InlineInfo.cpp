//===- InlineInfo.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/FileEntry.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/InlineInfo.h"
#include "llvm/Support/DataExtractor.h"
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

/// Decode an InlineInfo in Data at the specified offset.
///
/// A local helper function to decode InlineInfo objects. This function is
/// called recursively when parsing child InlineInfo objects.
///
/// \param Data The data extractor to decode from.
/// \param Offset The offset within \a Data to decode from.
/// \param BaseAddr The base address to use when decoding address ranges.
/// \returns An InlineInfo or an error describing the issue that was
/// encountered during decoding.
static llvm::Expected<InlineInfo> decode(DataExtractor &Data, uint64_t &Offset,
                                         uint64_t BaseAddr) {
  InlineInfo Inline;
  if (!Data.isValidOffset(Offset))
    return createStringError(std::errc::io_error,
        "0x%8.8" PRIx64 ": missing InlineInfo address ranges data", Offset);
  Inline.Ranges.decode(Data, BaseAddr, Offset);
  if (Inline.Ranges.empty())
    return Inline;
  if (!Data.isValidOffsetForDataOfSize(Offset, 1))
    return createStringError(std::errc::io_error,
        "0x%8.8" PRIx64 ": missing InlineInfo uint8_t indicating children",
        Offset);
  bool HasChildren = Data.getU8(&Offset) != 0;
  if (!Data.isValidOffsetForDataOfSize(Offset, 4))
    return createStringError(std::errc::io_error,
        "0x%8.8" PRIx64 ": missing InlineInfo uint32_t for name", Offset);
  Inline.Name = Data.getU32(&Offset);
  if (!Data.isValidOffset(Offset))
    return createStringError(std::errc::io_error,
        "0x%8.8" PRIx64 ": missing ULEB128 for InlineInfo call file", Offset);
  Inline.CallFile = (uint32_t)Data.getULEB128(&Offset);
  if (!Data.isValidOffset(Offset))
    return createStringError(std::errc::io_error,
        "0x%8.8" PRIx64 ": missing ULEB128 for InlineInfo call line", Offset);
  Inline.CallLine = (uint32_t)Data.getULEB128(&Offset);
  if (HasChildren) {
    // Child address ranges are encoded relative to the first address in the
    // parent InlineInfo object.
    const auto ChildBaseAddr = Inline.Ranges[0].Start;
    while (true) {
      llvm::Expected<InlineInfo> Child = decode(Data, Offset, ChildBaseAddr);
      if (!Child)
        return Child.takeError();
      // InlineInfo with empty Ranges termintes a child sibling chain.
      if (Child.get().Ranges.empty())
        break;
      Inline.Children.emplace_back(std::move(*Child));
    }
  }
  return Inline;
}

llvm::Expected<InlineInfo> InlineInfo::decode(DataExtractor &Data,
                                              uint64_t BaseAddr) {
  uint64_t Offset = 0;
  return ::decode(Data, Offset, BaseAddr);
}

llvm::Error InlineInfo::encode(FileWriter &O, uint64_t BaseAddr) const {
  // Users must verify the InlineInfo is valid prior to calling this funtion.
  // We don't want to emit any InlineInfo objects if they are not valid since
  // it will waste space in the GSYM file.
  if (!isValid())
    return createStringError(std::errc::invalid_argument,
                             "attempted to encode invalid InlineInfo object");
  Ranges.encode(O, BaseAddr);
  bool HasChildren = !Children.empty();
  O.writeU8(HasChildren);
  O.writeU32(Name);
  O.writeULEB(CallFile);
  O.writeULEB(CallLine);
  if (HasChildren) {
    // Child address ranges are encoded as relative to the first
    // address in the Ranges for this object. This keeps the offsets
    // small and allows for efficient encoding using ULEB offsets.
    const uint64_t ChildBaseAddr = Ranges[0].Start;
    for (const auto &Child : Children) {
      // Make sure all child address ranges are contained in the parent address
      // ranges.
      for (const auto &ChildRange: Child.Ranges) {
        if (!Ranges.contains(ChildRange))
          return createStringError(std::errc::invalid_argument,
                                   "child range not contained in parent");
      }
      llvm::Error Err = Child.encode(O, ChildBaseAddr);
      if (Err)
        return Err;
    }

    // Terminate child sibling chain by emitting a zero. This zero will cause
    // the decodeAll() function above to return false and stop the decoding
    // of child InlineInfo objects that are siblings.
    O.writeULEB(0);
  }
  return Error::success();
}
