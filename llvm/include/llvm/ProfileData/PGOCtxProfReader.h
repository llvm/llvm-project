//===--- PGOCtxProfReader.h - Contextual profile reader ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Reader for contextual iFDO profile, which comes in bitstream format.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_CTXINSTRPROFILEREADER_H
#define LLVM_PROFILEDATA_CTXINSTRPROFILEREADER_H

#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/ProfileData/PGOCtxProfWriter.h"
#include "llvm/Support/Error.h"
#include <map>

namespace llvm {
class PGOContextualProfile;
class PGOCtxProfContext;

namespace internal {
// When we traverse the contextual profile, we typically want to visit contexts
// pertaining to a specific function. To avoid traversing the whole tree, we
// want to keep a per-function list - which will be in preorder - of that
// function's contexts. This happens in PGOContextualProfile. For memory use
// efficiency, we want to make PGOCtxProfContext an intrusive double-linked list
// node. We need to handle the cases where PGOCtxProfContext nodes are moved and
// deleted: in both cases, we need to update the index (==list). We can do that
// directly from the node in the list, without knowing who the "parent" of the
// list is. That makes the ADT ilist overkill here. Finally, IndexNode is meant
// to be an implementation detail of PGOCtxProfContext, and the only reason it's
// factored out is to avoid implementing move semantics for all its members.
class IndexNode {
  // This class' members are intentionally private - it's a convenience
  // implementation detail.
  friend class ::llvm::PGOCtxProfContext;
  friend class ::llvm::PGOContextualProfile;

  IndexNode *Previous = nullptr;
  IndexNode *Next = nullptr;

  ~IndexNode() {
    if (Next)
      Next->Previous = Previous;
    if (Previous)
      Previous->Next = Next;
  }

  IndexNode(const IndexNode &Other) = delete;

  IndexNode(IndexNode &&Other) {
    // Copy the neighbor info
    Next = Other.Next;
    Previous = Other.Previous;

    // Update the neighbors to point to this object
    if (Other.Next)
      Other.Next->Previous = this;
    if (Other.Previous)
      Other.Previous->Next = this;

    // Make sure the dtor is a noop
    Other.Next = nullptr;
    Other.Previous = nullptr;
  }
  IndexNode() = default;
};
} // namespace internal

/// A node (context) in the loaded contextual profile, suitable for mutation
/// during IPO passes. We generally expect a fraction of counters and
/// callsites to be populated. We continue to model counters as vectors, but
/// callsites are modeled as a map of a map. The expectation is that, typically,
/// there is a small number of indirect targets (usually, 1 for direct calls);
/// but potentially a large number of callsites, and, as inlining progresses,
/// the callsite count of a caller will grow.
class PGOCtxProfContext final : public internal::IndexNode {
public:
  using CallTargetMapTy = std::map<GlobalValue::GUID, PGOCtxProfContext>;
  using CallsiteMapTy = std::map<uint32_t, CallTargetMapTy>;

private:
  friend class PGOCtxProfileReader;
  friend class PGOContextualProfile;

  GlobalValue::GUID GUID = 0;
  SmallVector<uint64_t, 16> Counters;
  CallsiteMapTy Callsites;

  PGOCtxProfContext(GlobalValue::GUID G, SmallVectorImpl<uint64_t> &&Counters)
      : GUID(G), Counters(std::move(Counters)) {}

  Expected<PGOCtxProfContext &>
  getOrEmplace(uint32_t Index, GlobalValue::GUID G,
               SmallVectorImpl<uint64_t> &&Counters);

  // Create a bogus context object, used for anchoring the index double linked
  // list - see IndexNode
  PGOCtxProfContext() = default;

public:
  PGOCtxProfContext(const PGOCtxProfContext &) = delete;
  PGOCtxProfContext &operator=(const PGOCtxProfContext &) = delete;
  PGOCtxProfContext(PGOCtxProfContext &&) = default;
  PGOCtxProfContext &operator=(PGOCtxProfContext &&) = delete;

  GlobalValue::GUID guid() const { return GUID; }
  const SmallVectorImpl<uint64_t> &counters() const { return Counters; }
  SmallVectorImpl<uint64_t> &counters() { return Counters; }

  uint64_t getEntrycount() const {
    assert(!Counters.empty() &&
           "Functions are expected to have at their entry BB instrumented, so "
           "there should always be at least 1 counter.");
    return Counters[0];
  }

  const CallsiteMapTy &callsites() const { return Callsites; }
  CallsiteMapTy &callsites() { return Callsites; }

  void ingestContext(uint32_t CSId, PGOCtxProfContext &&Other) {
    callsites()[CSId].emplace(Other.guid(), std::move(Other));
  }

  void ingestAllContexts(uint32_t CSId, CallTargetMapTy &&Other) {
    auto [_, Inserted] = callsites().try_emplace(CSId, std::move(Other));
    (void)Inserted;
    assert(Inserted &&
           "CSId was expected to be newly created as result of e.g. inlining");
  }

  void resizeCounters(uint32_t Size) { Counters.resize(Size); }

  bool hasCallsite(uint32_t I) const {
    return Callsites.find(I) != Callsites.end();
  }

  const CallTargetMapTy &callsite(uint32_t I) const {
    assert(hasCallsite(I) && "Callsite not found");
    return Callsites.find(I)->second;
  }

  CallTargetMapTy &callsite(uint32_t I) {
    assert(hasCallsite(I) && "Callsite not found");
    return Callsites.find(I)->second;
  }

  /// Insert this node's GUID as well as the GUIDs of the transitive closure of
  /// child nodes, into the provided set (technically, all that is required of
  /// `TSetOfGUIDs` is to have an `insert(GUID)` member)
  template <class TSetOfGUIDs>
  void getContainedGuids(TSetOfGUIDs &Guids) const {
    Guids.insert(GUID);
    for (const auto &[_, Callsite] : Callsites)
      for (const auto &[_, Callee] : Callsite)
        Callee.getContainedGuids(Guids);
  }
};

class PGOCtxProfileReader final {
  StringRef Magic;
  BitstreamCursor Cursor;
  Expected<BitstreamEntry> advance();
  Error readMetadata();
  Error wrongValue(const Twine &);
  Error unsupported(const Twine &);

  Expected<std::pair<std::optional<uint32_t>, PGOCtxProfContext>>
  readContext(bool ExpectIndex);
  bool canReadContext();

public:
  PGOCtxProfileReader(StringRef Buffer)
      : Magic(Buffer.substr(0, PGOCtxProfileWriter::ContainerMagic.size())),
        Cursor(Buffer.substr(PGOCtxProfileWriter::ContainerMagic.size())) {}

  Expected<std::map<GlobalValue::GUID, PGOCtxProfContext>> loadContexts();
};

void convertCtxProfToYaml(raw_ostream &OS,
                          const PGOCtxProfContext::CallTargetMapTy &);
} // namespace llvm
#endif
