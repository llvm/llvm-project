//===- PGOCtxProfReader.cpp - Contextual Instrumentation profile reader ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Read a contextual profile into a datastructure suitable for maintenance
// throughout IPO
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/PGOCtxProfReader.h"
#include "llvm/Bitstream/BitCodeEnums.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/PGOCtxProfWriter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/YAMLTraits.h"
#include <iterator>
#include <utility>

using namespace llvm;

// FIXME(#92054) - these Error handling macros are (re-)invented in a few
// places.
#define EXPECT_OR_RET(LHS, RHS)                                                \
  auto LHS = RHS;                                                              \
  if (!LHS)                                                                    \
    return LHS.takeError();

#define RET_ON_ERR(EXPR)                                                       \
  if (auto Err = (EXPR))                                                       \
    return Err;

Expected<PGOCtxProfContext &>
PGOCtxProfContext::getOrEmplace(uint32_t Index, GlobalValue::GUID G,
                                SmallVectorImpl<uint64_t> &&Counters) {
  auto [Iter, Inserted] =
      Callsites[Index].insert({G, PGOCtxProfContext(G, std::move(Counters))});
  if (!Inserted)
    return make_error<InstrProfError>(instrprof_error::invalid_prof,
                                      "Duplicate GUID for same callsite.");
  return Iter->second;
}

Expected<BitstreamEntry> PGOCtxProfileReader::advance() {
  return Cursor.advance(BitstreamCursor::AF_DontAutoprocessAbbrevs);
}

Error PGOCtxProfileReader::wrongValue(const Twine &Msg) {
  return make_error<InstrProfError>(instrprof_error::invalid_prof, Msg);
}

Error PGOCtxProfileReader::unsupported(const Twine &Msg) {
  return make_error<InstrProfError>(instrprof_error::unsupported_version, Msg);
}

bool PGOCtxProfileReader::canReadContext() {
  auto Blk = advance();
  if (!Blk) {
    consumeError(Blk.takeError());
    return false;
  }
  return Blk->Kind == BitstreamEntry::SubBlock &&
         Blk->ID == PGOCtxProfileBlockIDs::ContextNodeBlockID;
}

Expected<std::pair<std::optional<uint32_t>, PGOCtxProfContext>>
PGOCtxProfileReader::readContext(bool ExpectIndex) {
  RET_ON_ERR(Cursor.EnterSubBlock(PGOCtxProfileBlockIDs::ContextNodeBlockID));

  std::optional<ctx_profile::GUID> Guid;
  std::optional<SmallVector<uint64_t, 16>> Counters;
  std::optional<uint32_t> CallsiteIndex;

  SmallVector<uint64_t, 1> RecordValues;

  // We don't prescribe the order in which the records come in, and we are ok
  // if other unsupported records appear. We seek in the current subblock until
  // we get all we know.
  auto GotAllWeNeed = [&]() {
    return Guid.has_value() && Counters.has_value() &&
           (!ExpectIndex || CallsiteIndex.has_value());
  };
  while (!GotAllWeNeed()) {
    RecordValues.clear();
    EXPECT_OR_RET(Entry, advance());
    if (Entry->Kind != BitstreamEntry::Record)
      return wrongValue(
          "Expected records before encountering more subcontexts");
    EXPECT_OR_RET(ReadRecord,
                  Cursor.readRecord(bitc::UNABBREV_RECORD, RecordValues));
    switch (*ReadRecord) {
    case PGOCtxProfileRecords::Guid:
      if (RecordValues.size() != 1)
        return wrongValue("The GUID record should have exactly one value");
      Guid = RecordValues[0];
      break;
    case PGOCtxProfileRecords::Counters:
      Counters = std::move(RecordValues);
      if (Counters->empty())
        return wrongValue("Empty counters. At least the entry counter (one "
                          "value) was expected");
      break;
    case PGOCtxProfileRecords::CalleeIndex:
      if (!ExpectIndex)
        return wrongValue("The root context should not have a callee index");
      if (RecordValues.size() != 1)
        return wrongValue("The callee index should have exactly one value");
      CallsiteIndex = RecordValues[0];
      break;
    default:
      // OK if we see records we do not understand, like records (profile
      // components) introduced later.
      break;
    }
  }

  PGOCtxProfContext Ret(*Guid, std::move(*Counters));

  while (canReadContext()) {
    EXPECT_OR_RET(SC, readContext(true));
    auto &Targets = Ret.callsites()[*SC->first];
    auto [_, Inserted] =
        Targets.insert({SC->second.guid(), std::move(SC->second)});
    if (!Inserted)
      return wrongValue(
          "Unexpected duplicate target (callee) at the same callsite.");
  }
  return std::make_pair(CallsiteIndex, std::move(Ret));
}

Error PGOCtxProfileReader::readMetadata() {
  if (Magic.size() < PGOCtxProfileWriter::ContainerMagic.size() ||
      Magic != PGOCtxProfileWriter::ContainerMagic)
    return make_error<InstrProfError>(instrprof_error::invalid_prof,
                                      "Invalid magic");

  BitstreamEntry Entry;
  RET_ON_ERR(Cursor.advance().moveInto(Entry));
  if (Entry.Kind != BitstreamEntry::SubBlock ||
      Entry.ID != bitc::BLOCKINFO_BLOCK_ID)
    return unsupported("Expected Block ID");
  // We don't need the blockinfo to read the rest, it's metadata usable for e.g.
  // llvm-bcanalyzer.
  RET_ON_ERR(Cursor.SkipBlock());

  EXPECT_OR_RET(Blk, advance());
  if (Blk->Kind != BitstreamEntry::SubBlock)
    return unsupported("Expected Version record");
  RET_ON_ERR(
      Cursor.EnterSubBlock(PGOCtxProfileBlockIDs::ProfileMetadataBlockID));
  EXPECT_OR_RET(MData, advance());
  if (MData->Kind != BitstreamEntry::Record)
    return unsupported("Expected Version record");

  SmallVector<uint64_t, 1> Ver;
  EXPECT_OR_RET(Code, Cursor.readRecord(bitc::UNABBREV_RECORD, Ver));
  if (*Code != PGOCtxProfileRecords::Version)
    return unsupported("Expected Version record");
  if (Ver.size() != 1 || Ver[0] > PGOCtxProfileWriter::CurrentVersion)
    return unsupported("Version " + Twine(*Code) +
                       " is higher than supported version " +
                       Twine(PGOCtxProfileWriter::CurrentVersion));
  return Error::success();
}

Expected<std::map<GlobalValue::GUID, PGOCtxProfContext>>
PGOCtxProfileReader::loadContexts() {
  std::map<GlobalValue::GUID, PGOCtxProfContext> Ret;
  RET_ON_ERR(readMetadata());
  while (canReadContext()) {
    EXPECT_OR_RET(E, readContext(false));
    auto Key = E->second.guid();
    if (!Ret.insert({Key, std::move(E->second)}).second)
      return wrongValue("Duplicate roots");
  }
  return std::move(Ret);
}

namespace {
// We want to pass `const` values PGOCtxProfContext references to the yaml
// converter, and the regular yaml mapping APIs are designed to handle both
// serialization and deserialization, which prevents using const for
// serialization. Using an intermediate datastructure is overkill, both
// space-wise and design complexity-wise. Instead, we use the lower-level APIs.
void toYaml(yaml::Output &Out, const PGOCtxProfContext &Ctx);

void toYaml(yaml::Output &Out,
            const PGOCtxProfContext::CallTargetMapTy &CallTargets) {
  Out.beginSequence();
  size_t Index = 0;
  void *SaveData = nullptr;
  for (const auto &[_, Ctx] : CallTargets) {
    Out.preflightElement(Index++, SaveData);
    toYaml(Out, Ctx);
    Out.postflightElement(nullptr);
  }
  Out.endSequence();
}

void toYaml(yaml::Output &Out,
            const PGOCtxProfContext::CallsiteMapTy &Callsites) {
  auto AllCS = ::llvm::make_first_range(Callsites);
  auto MaxIt = ::llvm::max_element(AllCS);
  assert(MaxIt != AllCS.end() && "We should have a max value because the "
                                 "callsites collection is not empty.");
  void *SaveData = nullptr;
  Out.beginSequence();
  for (auto I = 0U; I <= *MaxIt; ++I) {
    Out.preflightElement(I, SaveData);
    auto It = Callsites.find(I);
    if (It == Callsites.end()) {
      // This will produce a `[ ]` sequence, which is what we want here.
      Out.beginFlowSequence();
      Out.endFlowSequence();
    } else {
      toYaml(Out, It->second);
    }
    Out.postflightElement(nullptr);
  }
  Out.endSequence();
}

void toYaml(yaml::Output &Out, const PGOCtxProfContext &Ctx) {
  yaml::EmptyContext Empty;
  Out.beginMapping();
  void *SaveInfo = nullptr;
  bool UseDefault = false;
  {
    Out.preflightKey("Guid", /*Required=*/true, /*SameAsDefault=*/false,
                     UseDefault, SaveInfo);
    auto Guid = Ctx.guid();
    yaml::yamlize(Out, Guid, true, Empty);
    Out.postflightKey(nullptr);
  }
  {
    Out.preflightKey("Counters", true, false, UseDefault, SaveInfo);
    Out.beginFlowSequence();
    for (size_t I = 0U, E = Ctx.counters().size(); I < E; ++I) {
      Out.preflightFlowElement(I, SaveInfo);
      uint64_t V = Ctx.counters()[I];
      yaml::yamlize(Out, V, true, Empty);
      Out.postflightFlowElement(SaveInfo);
    }
    Out.endFlowSequence();
    Out.postflightKey(nullptr);
  }
  if (!Ctx.callsites().empty()) {
    Out.preflightKey("Callsites", true, false, UseDefault, SaveInfo);
    toYaml(Out, Ctx.callsites());
    Out.postflightKey(nullptr);
  }
  Out.endMapping();
}
} // namespace

void llvm::convertCtxProfToYaml(
    raw_ostream &OS, const PGOCtxProfContext::CallTargetMapTy &Profiles) {
  yaml::Output Out(OS);
  toYaml(Out, Profiles);
}