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
#include "llvm/ProfileData/CtxInstrContextNode.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

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
  const auto Version = VersionOverride.value_or(CurrentVersion);
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

namespace {

/// Representation of the context node suitable for yaml serialization /
/// deserialization.
struct SerializableCtxRepresentation {
  ctx_profile::GUID Guid = 0;
  std::vector<uint64_t> Counters;
  std::vector<std::vector<SerializableCtxRepresentation>> Callsites;
};

ctx_profile::ContextNode *
createNode(std::vector<std::unique_ptr<char[]>> &Nodes,
           const std::vector<SerializableCtxRepresentation> &DCList);

// Convert a DeserializableCtx into a ContextNode, potentially linking it to
// its sibling (e.g. callee at same callsite) "Next".
ctx_profile::ContextNode *
createNode(std::vector<std::unique_ptr<char[]>> &Nodes,
           const SerializableCtxRepresentation &DC,
           ctx_profile::ContextNode *Next = nullptr) {
  auto AllocSize = ctx_profile::ContextNode::getAllocSize(DC.Counters.size(),
                                                          DC.Callsites.size());
  auto *Mem = Nodes.emplace_back(std::make_unique<char[]>(AllocSize)).get();
  std::memset(Mem, 0, AllocSize);
  auto *Ret = new (Mem) ctx_profile::ContextNode(DC.Guid, DC.Counters.size(),
                                                 DC.Callsites.size(), Next);
  std::memcpy(Ret->counters(), DC.Counters.data(),
              sizeof(uint64_t) * DC.Counters.size());
  for (const auto &[I, DCList] : llvm::enumerate(DC.Callsites))
    Ret->subContexts()[I] = createNode(Nodes, DCList);
  return Ret;
}

// Convert a list of SerializableCtxRepresentation into a linked list of
// ContextNodes.
ctx_profile::ContextNode *
createNode(std::vector<std::unique_ptr<char[]>> &Nodes,
           const std::vector<SerializableCtxRepresentation> &DCList) {
  ctx_profile::ContextNode *List = nullptr;
  for (const auto &DC : DCList)
    List = createNode(Nodes, DC, List);
  return List;
}
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(SerializableCtxRepresentation)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<SerializableCtxRepresentation>)
template <> struct yaml::MappingTraits<SerializableCtxRepresentation> {
  static void mapping(yaml::IO &IO, SerializableCtxRepresentation &SCR) {
    IO.mapRequired("Guid", SCR.Guid);
    IO.mapRequired("Counters", SCR.Counters);
    IO.mapOptional("Callsites", SCR.Callsites);
  }
};

Error llvm::createCtxProfFromYAML(StringRef Profile, raw_ostream &Out) {
  yaml::Input In(Profile);
  std::vector<SerializableCtxRepresentation> DCList;
  In >> DCList;
  if (In.error())
    return createStringError(In.error(), "incorrect yaml content");
  std::vector<std::unique_ptr<char[]>> Nodes;
  std::error_code EC;
  if (EC)
    return createStringError(EC, "failed to open output");
  PGOCtxProfileWriter Writer(Out);
  for (const auto &DC : DCList) {
    auto *TopList = createNode(Nodes, DC);
    if (!TopList)
      return createStringError(
          "Unexpected error converting internal structure to ctx profile");
    Writer.write(*TopList);
  }
  if (EC)
    return createStringError(EC, "failed to write output");
  return Error::success();
}
