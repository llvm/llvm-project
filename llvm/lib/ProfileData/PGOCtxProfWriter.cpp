//===- PGOCtxProfWriter.cpp - Contextual Instrumentation profile writer ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Write a contextual profile to bitstream.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/PGOCtxProfWriter.h"
#include "llvm/Bitstream/BitCodeEnums.h"

using namespace llvm;
using namespace llvm::ctx_profile;

PGOCtxProfileWriter::PGOCtxProfileWriter(
    raw_ostream &Out, std::optional<unsigned> VersionOverride)
    : Writer(Out, 0) {
  static_assert(ContainerMagic.size() == 4);
  Out.write(ContainerMagic.data(), ContainerMagic.size());
  Writer.EnterBlockInfoBlock();
  {
    auto DescribeBlock = [&](unsigned ID, StringRef Name) {
      Writer.EmitRecord(bitc::BLOCKINFO_CODE_SETBID,
                        SmallVector<unsigned, 1>{ID});
      Writer.EmitRecord(bitc::BLOCKINFO_CODE_BLOCKNAME,
                        llvm::arrayRefFromStringRef(Name));
    };
    SmallVector<uint64_t, 16> Data;
    auto DescribeRecord = [&](unsigned RecordID, StringRef Name) {
      Data.clear();
      Data.push_back(RecordID);
      llvm::append_range(Data, Name);
      Writer.EmitRecord(bitc::BLOCKINFO_CODE_SETRECORDNAME, Data);
    };
    DescribeBlock(PGOCtxProfileBlockIDs::ProfileMetadataBlockID, "Metadata");
    DescribeRecord(PGOCtxProfileRecords::Version, "Version");
    DescribeBlock(PGOCtxProfileBlockIDs::ContextNodeBlockID, "Context");
    DescribeRecord(PGOCtxProfileRecords::Guid, "GUID");
    DescribeRecord(PGOCtxProfileRecords::CalleeIndex, "CalleeIndex");
    DescribeRecord(PGOCtxProfileRecords::Counters, "Counters");
  }
  Writer.ExitBlock();
  Writer.EnterSubblock(PGOCtxProfileBlockIDs::ProfileMetadataBlockID, CodeLen);
  const auto Version = VersionOverride ? *VersionOverride : CurrentVersion;
  Writer.EmitRecord(PGOCtxProfileRecords::Version,
                    SmallVector<unsigned, 1>({Version}));
}

void PGOCtxProfileWriter::writeCounters(const ContextNode &Node) {
  Writer.EmitCode(bitc::UNABBREV_RECORD);
  Writer.EmitVBR(PGOCtxProfileRecords::Counters, VBREncodingBits);
  Writer.EmitVBR(Node.counters_size(), VBREncodingBits);
  for (uint32_t I = 0U; I < Node.counters_size(); ++I)
    Writer.EmitVBR64(Node.counters()[I], VBREncodingBits);
}

// recursively write all the subcontexts. We do need to traverse depth first to
// model the context->subcontext implicitly, and since this captures call
// stacks, we don't really need to be worried about stack overflow and we can
// keep the implementation simple.
void PGOCtxProfileWriter::writeImpl(std::optional<uint32_t> CallerIndex,
                                    const ContextNode &Node) {
  Writer.EnterSubblock(PGOCtxProfileBlockIDs::ContextNodeBlockID, CodeLen);
  Writer.EmitRecord(PGOCtxProfileRecords::Guid,
                    SmallVector<uint64_t, 1>{Node.guid()});
  if (CallerIndex)
    Writer.EmitRecord(PGOCtxProfileRecords::CalleeIndex,
                      SmallVector<uint64_t, 1>{*CallerIndex});
  writeCounters(Node);
  for (uint32_t I = 0U; I < Node.callsites_size(); ++I)
    for (const auto *Subcontext = Node.subContexts()[I]; Subcontext;
         Subcontext = Subcontext->next())
      writeImpl(I, *Subcontext);
  Writer.ExitBlock();
}

void PGOCtxProfileWriter::write(const ContextNode &RootNode) {
  writeImpl(std::nullopt, RootNode);
}
