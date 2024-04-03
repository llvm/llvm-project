//===--- CtxInstrProfileReader.h - Ctx iFDO profile reader ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Reader for contextual iFDO profile, in bitstream format.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_CTXINSTRPROFILEREADER_H
#define LLVM_PROFILEDATA_CTXINSTRPROFILEREADER_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include <map>
#include <vector>

namespace llvm {
class ContextualProfile final {
  friend class ContextualInstrProfReader;
  GlobalValue::GUID GUID = 0;
  SmallVector<uint64_t, 16> Counters;
  std::vector<std::map<GlobalValue::GUID, ContextualProfile>> Callsites;

  ContextualProfile(GlobalValue::GUID G, SmallVectorImpl<uint64_t> &&Counters)
      : GUID(G), Counters(std::move(Counters)) {}

  Expected<ContextualProfile &>
  getOrEmplace(uint32_t Index, GlobalValue::GUID G,
               SmallVectorImpl<uint64_t> &&Counters) {
    if (Callsites.size() <= Index)
      Callsites.resize(Index + 1);
    auto I =
        Callsites[Index].insert({G, ContextualProfile(G, std::move(Counters))});
    if (!I.second)
      return make_error<StringError>(llvm::errc::invalid_argument,
                                     "Duplicate GUID for same callsite.");
    return I.first->second;
  }

public:
  ContextualProfile(const ContextualProfile &) = delete;
  ContextualProfile &operator=(const ContextualProfile &) = delete;
  ContextualProfile(ContextualProfile &&) = default;
  ContextualProfile &operator=(ContextualProfile &&) = default;

  GlobalValue::GUID guid() const { return GUID; }
  const SmallVector<uint64_t, 16> &counters() const { return Counters; }
  const std::vector<std::map<GlobalValue::GUID, ContextualProfile>> &
  callsites() const {
    return Callsites;
  }

  void getContainedGuids(DenseSet<GlobalValue::GUID> &Guids) const {
    Guids.insert(GUID);
    for (const auto &Callsite : Callsites)
      for (const auto &[_, Callee] : Callsite)
        Callee.getContainedGuids(Guids);
  }
};

class ContextualInstrProfReader final {
  enum Codes {
    Invalid,
    Guid,
    CalleeIndex,
    Counters,
  };

  BitstreamCursor Cursor;

  struct ContextData {
    GlobalValue::GUID GUID;
    std::optional<uint32_t> Index;
    SmallVector<uint64_t, 16> Counters;
  };
  Expected<unsigned>
  readUnabbrevRecord(SmallVectorImpl<uint64_t> &Vals,
                     std::optional<Codes> ExpectedCode = std::nullopt) {
    auto Code = Cursor.ReadCode();
    if (!Code)
      return Code.takeError();
    if (*Code != bitc::UNABBREV_RECORD)
      return make_error<StringError>(llvm::errc::invalid_argument,
                                     "Invalid code.");
    auto Record = Cursor.readRecord(bitc::UNABBREV_RECORD, Vals);
    if (!Record)
      return Record.takeError();
    if (!ExpectedCode)
      return *Record;
    if (*Record != *ExpectedCode)
      return make_error<StringError>(llvm::errc::invalid_argument,
                                     "Unexpected code.");
    return *Record;
  }

  Expected<ContextData> readContextData() {
    ContextData Ret;
    SmallVector<uint64_t, 1> Data64;
    auto GuidRec = readUnabbrevRecord(Data64, Codes::Guid);
    if (!GuidRec)
      return GuidRec.takeError();
    Ret.GUID = Data64[0];
    auto IndexOrCounters = readUnabbrevRecord(Ret.Counters);
    if (!IndexOrCounters)
      return IndexOrCounters.takeError();

    if (*IndexOrCounters == Codes::CalleeIndex) {
      Ret.Index = Ret.Counters[0];
      Ret.Counters.clear();
      auto NextRecord = readUnabbrevRecord(Ret.Counters, Codes::Counters);
      if (!NextRecord)
        return NextRecord.takeError();
    } else if (*IndexOrCounters != Codes::Counters) {
      return make_error<StringError>(llvm::errc::invalid_argument,
                                     "Expected counters.");
    }
    return Ret;
  }

  Error failIfCannotEnterSubBlock() {
    auto MaybeEntry =
        Cursor.advance(BitstreamCursor::AF_DontAutoprocessAbbrevs);
    if (!MaybeEntry)
      return MaybeEntry.takeError();
    if (MaybeEntry->Kind != BitstreamEntry::SubBlock)
      return make_error<StringError>(llvm::errc::invalid_argument,
                                     "Expected a subblock.");
    if (MaybeEntry->ID != 100)
      return make_error<StringError>(llvm::errc::invalid_argument,
                                     "Expected subblock ID 100.");
    if (auto Err = Cursor.EnterSubBlock(MaybeEntry->ID))
      return Err;
    return Error::success();
  }

  Error failIfCannotReadSubContexts(ContextualProfile &Parent) {
    while (!failIfCannotEnterSubBlock()) {
      auto Ctx = readContextData();
      if (!Ctx)
        return Ctx.takeError();
      if (!Ctx->Index)
        return make_error<StringError>(
            llvm::errc::invalid_argument,
            "Invalid subcontext: should have an index.");
      auto P =
          Parent.getOrEmplace(*Ctx->Index, Ctx->GUID, std::move(Ctx->Counters));
      if (!P)
        return P.takeError();
      auto Sub = failIfCannotReadSubContexts(*P);
      if (Sub)
        return Sub;
    }
    return Error::success();
  }

public:
  ContextualInstrProfReader(StringRef ProfileFile) : Cursor(ProfileFile) {}

  Expected<std::map<GlobalValue::GUID, ContextualProfile>> loadContexts() {
    auto MaybeMagic = Cursor.Read(32);
    if (!MaybeMagic)
      return MaybeMagic.takeError();
    if (*MaybeMagic != 0xfafababa)
      return make_error<StringError>(llvm::errc::invalid_argument,
                                     "Invalid magic.");
    std::map<GlobalValue::GUID, ContextualProfile> Ret;
    while (!failIfCannotEnterSubBlock()) {
      auto Ctx = readContextData();
      if (!Ctx)
        return Ctx.takeError();
      if (Ctx->Index)
        return make_error<StringError>(llvm::errc::invalid_argument,
                                       "Invalid root: should have no index.");
      auto Ins = Ret.insert(
          {Ctx->GUID, ContextualProfile(Ctx->GUID, std::move(Ctx->Counters))});
      if (!Ins.second)
        return make_error<StringError>(llvm::errc::invalid_argument,
                                       "Duplicate GUID for same root.");
      auto ReadRest = failIfCannotReadSubContexts(Ins.first->second);
      if (ReadRest)
        return ReadRest;
    }
    return Ret;
  }
};
} // namespace llvm
#endif