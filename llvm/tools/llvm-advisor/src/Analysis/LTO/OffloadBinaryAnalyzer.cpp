//===--- OffloadBinaryAnalyzer.cpp - LLVM Advisor ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/LTO/OffloadBinaryAnalyzer.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm::advisor {

Expected<std::unique_ptr<CapabilityResult>>
OffloadBinaryAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  if (Context.ObjectPath.empty() || !sys::fs::exists(Context.ObjectPath))
    return makeUnavailableResult(CapID, UnitID,
                                 "no object file for offload binary inspection");

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(Context.ObjectPath);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(), "cannot read %s",
                             Context.ObjectPath.c_str());

  SmallVector<object::OffloadFile, 4> Binaries;
  if (Error Err = object::extractOffloadBinaries((*BufOrErr)->getMemBufferRef(),
                                                 Binaries))
    return std::move(Err);

  if (Binaries.empty()) {
    return makeJSONResult(CapID, UnitID, json::Object{
        {"object_path", Context.ObjectPath},
        {"has_offload_sections", false},
        {"entry_count", 0},
        {"entries", json::Array{}},
    });
  }

  // Group by offload kind for summary.
  StringMap<int64_t> KindCounts;
  json::Array Entries;

  for (const object::OffloadFile &OF : Binaries) {
    const object::OffloadBinary *OB = OF.getBinary();
    if (!OB)
      continue;

    StringRef KindName = object::getOffloadKindName(OB->getOffloadKind());
    KindCounts[KindName]++;

    json::Object Entry;
    Entry["offload_kind"] = KindName;
    Entry["image_kind"] = object::getImageKindName(OB->getImageKind());
    Entry["triple"] = OB->getTriple();
    Entry["arch"] = OB->getArch();
    Entry["image_size"] = static_cast<int64_t>(OB->getImage().size());

    // Collect all string metadata (triple, arch, producer, etc.).
    json::Object Strings;
    for (auto [K, V] : OB->strings())
      Strings[K] = V;
    if (!Strings.empty())
      Entry["metadata"] = std::move(Strings);

    Entries.push_back(std::move(Entry));
  }

  json::Object KindSummary;
  for (auto &KV : KindCounts)
    KindSummary[KV.getKey()] = KV.second;

  return makeJSONResult(CapID, UnitID, json::Object{
      {"object_path", Context.ObjectPath},
      {"has_offload_sections", true},
      {"entry_count", static_cast<int64_t>(Binaries.size())},
      {"by_offload_kind", std::move(KindSummary)},
      {"entries", std::move(Entries)},
  });
}

} // namespace llvm::advisor
