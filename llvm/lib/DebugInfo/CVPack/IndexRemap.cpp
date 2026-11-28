//===- IndexRemap.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CVPack/IndexRemap.h"
#include "CVPackUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;
using namespace llvm::support::endian;

void IndexRemap::add(uint64_t From, uint64_t Count, uint64_t To) {
  Ranges.push_back({static_cast<uint32_t>(From), static_cast<uint32_t>(Count),
                    static_cast<uint32_t>(To)});
}

Error IndexRemap::finalize() {
  llvm::sort(Ranges,
             [](const Range &A, const Range &B) { return A.From < B.From; });
  for (size_t I = 1; I < Ranges.size(); ++I)
    if (uint64_t(Ranges[I - 1].From) + Ranges[I - 1].Count > Ranges[I].From)
      return malformedRecord("object maps one pack index more than once");
  return Error::success();
}

std::optional<uint32_t> IndexRemap::lookup(uint32_t Index) const {
  auto It = llvm::upper_bound(
      Ranges, Index, [](uint32_t V, const Range &R) { return V < R.From; });
  if (It == Ranges.begin())
    return std::nullopt;
  --It;
  if (Index - It->From >= It->Count)
    return std::nullopt;
  return It->To + (Index - It->From);
}

Error llvm::codeview::cvpack::remapTypeIndices(MutableArrayRef<uint8_t> Record,
                                               bool IsSymbol,
                                               const IndexRemap &Map) {
  return forEachTypeIndex(
      Record, IsSymbol, [&](uint32_t Offset, TiRefKind) -> Error {
        uint32_t Index = read32le(Record.data() + Offset);
        if (Index < TypeIndex::FirstNonSimpleIndex)
          return Error::success();
        std::optional<uint32_t> Local = Map.lookup(Index);
        if (!Local)
          return malformedRecord(
              "pack type reference 0x" + Twine::utohexstr(Index) + " of " +
              (IsSymbol ? "symbol" : "type") + " record kind 0x" +
              Twine::utohexstr(read16le(Record.data() + 2)) +
              " is absent from map");
        support::endian::write32le(Record.data() + Offset, *Local);
        return Error::success();
      });
}
