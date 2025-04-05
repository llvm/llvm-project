//===- PGOCtxProfWriter.h - Contextual Profile Writer -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a utility for writing a contextual profile to bitstream.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_PGOCTXPROFWRITER_H_
#define LLVM_PROFILEDATA_PGOCTXPROFWRITER_H_

#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitstream/BitCodeEnums.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/ProfileData/CtxInstrContextNode.h"

namespace llvm {
enum PGOCtxProfileRecords {
  Invalid = 0,
  Version,
  Guid,
  CallsiteIndex,
  Counters,
  TotalRootEntryCount
};

enum PGOCtxProfileBlockIDs {
  FIRST_VALID = bitc::FIRST_APPLICATION_BLOCKID,
  ProfileMetadataBlockID = FIRST_VALID,
  ContextsSectionBlockID = ProfileMetadataBlockID + 1,
  ContextRootBlockID = ContextsSectionBlockID + 1,
  ContextNodeBlockID = ContextRootBlockID + 1,
  FlatProfilesSectionBlockID = ContextNodeBlockID + 1,
  FlatProfileBlockID = FlatProfilesSectionBlockID + 1,
  UnhandledBlockID = FlatProfileBlockID + 1,
  LAST_VALID = UnhandledBlockID
};

/// Write one or more ContextNodes to the provided raw_fd_stream.
/// The caller must destroy the PGOCtxProfileWriter object before closing the
/// stream.
/// The design allows serializing a bunch of contexts embedded in some other
/// file. The overall format is:
///
///  [... other data written to the stream...]
///  SubBlock(ProfileMetadataBlockID)
///   Version
///   SubBlock(ContextNodeBlockID)
///     [RECORDS]
///     SubBlock(ContextNodeBlockID)
///       [RECORDS]
///       [... more SubBlocks]
///     EndBlock
///   EndBlock
///
/// The "RECORDS" are bitsream records. The IDs are in CtxProfileCodes (except)
/// for Version, which is just for metadata). All contexts will have Guid and
/// Counters, and all but the roots have CalleeIndex. The order in which the
/// records appear does not matter, but they must precede any subcontexts,
/// because that helps keep the reader code simpler.
///
/// Subblock containment captures the context->subcontext relationship. The
/// "next()" relationship in the raw profile, between call targets of indirect
/// calls, are just modeled as peer subblocks where the callee index is the
/// same.
///
/// Versioning: the writer may produce additional records not known by the
/// reader. The version number indicates a more structural change.
/// The current version, in particular, is set up to expect optional extensions
/// like value profiling - which would appear as additional records. For
/// example, value profiling would produce a new record with a new record ID,
/// containing the profiled values (much like the counters)
class PGOCtxProfileWriter final : public ctx_profile::ProfileWriter {
  enum class EmptyContextCriteria { None, EntryIsZero, AllAreZero };

  BitstreamWriter Writer;
  const bool IncludeEmpty;

  void writeGuid(ctx_profile::GUID Guid);
  void writeCallsiteIndex(uint32_t Index);
  void writeRootEntryCount(uint64_t EntryCount);
  void writeCounters(ArrayRef<uint64_t> Counters);
  void writeNode(uint32_t CallerIndex, const ctx_profile::ContextNode &Node);
  void writeSubcontexts(const ctx_profile::ContextNode &Node);

public:
  PGOCtxProfileWriter(raw_ostream &Out,
                      std::optional<unsigned> VersionOverride = std::nullopt,
                      bool IncludeEmpty = false);
  ~PGOCtxProfileWriter() { Writer.ExitBlock(); }

  void startContextSection() override;
  void writeContextual(const ctx_profile::ContextNode &RootNode,
                       const ctx_profile::ContextNode *Unhandled,
                       uint64_t TotalRootEntryCount) override;
  void endContextSection() override;

  void startFlatSection() override;
  void writeFlat(ctx_profile::GUID Guid, const uint64_t *Buffer,
                 size_t BufferSize) override;
  void endFlatSection() override;

  // constants used in writing which a reader may find useful.
  static constexpr unsigned CodeLen = 2;
  static constexpr uint32_t CurrentVersion = 4;
  static constexpr unsigned VBREncodingBits = 6;
  static constexpr StringRef ContainerMagic = "CTXP";
};

Error createCtxProfFromYAML(StringRef Profile, raw_ostream &Out);
} // namespace llvm
#endif
