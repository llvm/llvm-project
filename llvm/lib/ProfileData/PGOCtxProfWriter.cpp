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
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::ctx_profile;

static cl::opt<bool>
    IncludeEmptyOpt("ctx-prof-include-empty", cl::init(false),
                    cl::desc("Also write profiles with all-zero counters. "
                             "Intended for testing/debugging."));

PGOCtxProfileWriter::PGOCtxProfileWriter(
    raw_ostream &Out, std::optional<unsigned> VersionOverride,
    bool IncludeEmpty)
    : Writer(Out, 0),
      IncludeEmpty(IncludeEmptyOpt.getNumOccurrences() > 0 ? IncludeEmptyOpt
                                                           : IncludeEmpty) {
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
    DescribeBlock(PGOCtxProfileBlockIDs::ContextsSectionBlockID, "Contexts");
    DescribeBlock(PGOCtxProfileBlockIDs::ContextRootBlockID, "Root");
    DescribeRecord(PGOCtxProfileRecords::Guid, "GUID");
    DescribeRecord(PGOCtxProfileRecords::TotalRootEntryCount,
                   "TotalRootEntryCount");
    DescribeRecord(PGOCtxProfileRecords::Counters, "Counters");
    DescribeBlock(PGOCtxProfileBlockIDs::ContextNodeBlockID, "Context");
    DescribeRecord(PGOCtxProfileRecords::Guid, "GUID");
    DescribeRecord(PGOCtxProfileRecords::CallsiteIndex, "CalleeIndex");
    DescribeRecord(PGOCtxProfileRecords::Counters, "Counters");
    DescribeBlock(PGOCtxProfileBlockIDs::FlatProfilesSectionBlockID,
                  "FlatProfiles");
    DescribeBlock(PGOCtxProfileBlockIDs::FlatProfileBlockID, "Flat");
    DescribeRecord(PGOCtxProfileRecords::Guid, "GUID");
    DescribeRecord(PGOCtxProfileRecords::Counters, "Counters");
  }
  Writer.ExitBlock();
  Writer.EnterSubblock(PGOCtxProfileBlockIDs::ProfileMetadataBlockID, CodeLen);
  const auto Version = VersionOverride.value_or(CurrentVersion);
  Writer.EmitRecord(PGOCtxProfileRecords::Version,
                    SmallVector<unsigned, 1>({Version}));
}

void PGOCtxProfileWriter::writeCounters(ArrayRef<uint64_t> Counters) {
  Writer.EmitCode(bitc::UNABBREV_RECORD);
  Writer.EmitVBR(PGOCtxProfileRecords::Counters, VBREncodingBits);
  Writer.EmitVBR(Counters.size(), VBREncodingBits);
  for (uint64_t C : Counters)
    Writer.EmitVBR64(C, VBREncodingBits);
}

void PGOCtxProfileWriter::writeGuid(ctx_profile::GUID Guid) {
  Writer.EmitRecord(PGOCtxProfileRecords::Guid, SmallVector<uint64_t, 1>{Guid});
}

void PGOCtxProfileWriter::writeCallsiteIndex(uint32_t CallsiteIndex) {
  Writer.EmitRecord(PGOCtxProfileRecords::CallsiteIndex,
                    SmallVector<uint64_t, 1>{CallsiteIndex});
}

void PGOCtxProfileWriter::writeRootEntryCount(uint64_t TotalRootEntryCount) {
  Writer.EmitRecord(PGOCtxProfileRecords::TotalRootEntryCount,
                    SmallVector<uint64_t, 1>{TotalRootEntryCount});
}

// recursively write all the subcontexts. We do need to traverse depth first to
// model the context->subcontext implicitly, and since this captures call
// stacks, we don't really need to be worried about stack overflow and we can
// keep the implementation simple.
void PGOCtxProfileWriter::writeNode(uint32_t CallsiteIndex,
                                    const ContextNode &Node) {
  // A node with no counters is an error. We don't expect this to happen from
  // the runtime, rather, this is interesting for testing the reader.
  if (!IncludeEmpty && (Node.counters_size() > 0 && Node.entrycount() == 0))
    return;
  Writer.EnterSubblock(PGOCtxProfileBlockIDs::ContextNodeBlockID, CodeLen);
  writeGuid(Node.guid());
  writeCallsiteIndex(CallsiteIndex);
  writeCounters({Node.counters(), Node.counters_size()});
  writeSubcontexts(Node);
  Writer.ExitBlock();
}

void PGOCtxProfileWriter::writeSubcontexts(const ContextNode &Node) {
  for (uint32_t I = 0U; I < Node.callsites_size(); ++I)
    for (const auto *Subcontext = Node.subContexts()[I]; Subcontext;
         Subcontext = Subcontext->next())
      writeNode(I, *Subcontext);
}

void PGOCtxProfileWriter::startContextSection() {
  Writer.EnterSubblock(PGOCtxProfileBlockIDs::ContextsSectionBlockID, CodeLen);
}

void PGOCtxProfileWriter::startFlatSection() {
  Writer.EnterSubblock(PGOCtxProfileBlockIDs::FlatProfilesSectionBlockID,
                       CodeLen);
}

void PGOCtxProfileWriter::endContextSection() { Writer.ExitBlock(); }
void PGOCtxProfileWriter::endFlatSection() { Writer.ExitBlock(); }

void PGOCtxProfileWriter::writeContextual(const ContextNode &RootNode,
                                          uint64_t TotalRootEntryCount) {
  if (!IncludeEmpty && (!TotalRootEntryCount || (RootNode.counters_size() > 0 &&
                                                 RootNode.entrycount() == 0)))
    return;
  Writer.EnterSubblock(PGOCtxProfileBlockIDs::ContextRootBlockID, CodeLen);
  writeGuid(RootNode.guid());
  writeRootEntryCount(TotalRootEntryCount);
  writeCounters({RootNode.counters(), RootNode.counters_size()});
  writeSubcontexts(RootNode);
  Writer.ExitBlock();
}

void PGOCtxProfileWriter::writeFlat(ctx_profile::GUID Guid,
                                    const uint64_t *Buffer, size_t Size) {
  Writer.EnterSubblock(PGOCtxProfileBlockIDs::FlatProfileBlockID, CodeLen);
  writeGuid(Guid);
  writeCounters({Buffer, Size});
  Writer.ExitBlock();
}

namespace {

/// Representation of the context node suitable for yaml serialization /
/// deserialization.
struct SerializableCtxRepresentation {
  ctx_profile::GUID Guid = 0;
  std::vector<uint64_t> Counters;
  std::vector<std::vector<SerializableCtxRepresentation>> Callsites;
};

struct SerializableRootRepresentation : public SerializableCtxRepresentation {
  uint64_t TotalRootEntryCount = 0;
};

using SerializableFlatProfileRepresentation =
    std::pair<ctx_profile::GUID, std::vector<uint64_t>>;

struct SerializableProfileRepresentation {
  std::vector<SerializableRootRepresentation> Contexts;
  std::vector<SerializableFlatProfileRepresentation> FlatProfiles;
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
LLVM_YAML_IS_SEQUENCE_VECTOR(SerializableRootRepresentation)
LLVM_YAML_IS_SEQUENCE_VECTOR(SerializableFlatProfileRepresentation)
template <> struct yaml::MappingTraits<SerializableCtxRepresentation> {
  static void mapping(yaml::IO &IO, SerializableCtxRepresentation &SCR) {
    IO.mapRequired("Guid", SCR.Guid);
    IO.mapRequired("Counters", SCR.Counters);
    IO.mapOptional("Callsites", SCR.Callsites);
  }
};

template <> struct yaml::MappingTraits<SerializableRootRepresentation> {
  static void mapping(yaml::IO &IO, SerializableRootRepresentation &R) {
    yaml::MappingTraits<SerializableCtxRepresentation>::mapping(IO, R);
    IO.mapRequired("TotalRootEntryCount", R.TotalRootEntryCount);
  }
};

template <> struct yaml::MappingTraits<SerializableProfileRepresentation> {
  static void mapping(yaml::IO &IO, SerializableProfileRepresentation &SPR) {
    IO.mapOptional("Contexts", SPR.Contexts);
    IO.mapOptional("FlatProfiles", SPR.FlatProfiles);
  }
};

template <> struct yaml::MappingTraits<SerializableFlatProfileRepresentation> {
  static void mapping(yaml::IO &IO,
                      SerializableFlatProfileRepresentation &SFPR) {
    IO.mapRequired("Guid", SFPR.first);
    IO.mapRequired("Counters", SFPR.second);
  }
};

Error llvm::createCtxProfFromYAML(StringRef Profile, raw_ostream &Out) {
  yaml::Input In(Profile);
  SerializableProfileRepresentation SPR;
  In >> SPR;
  if (In.error())
    return createStringError(In.error(), "incorrect yaml content");
  std::vector<std::unique_ptr<char[]>> Nodes;
  std::error_code EC;
  if (EC)
    return createStringError(EC, "failed to open output");
  PGOCtxProfileWriter Writer(Out);

  if (!SPR.Contexts.empty()) {
    Writer.startContextSection();
    for (const auto &DC : SPR.Contexts) {
      auto *TopList = createNode(Nodes, DC);
      if (!TopList)
        return createStringError(
            "Unexpected error converting internal structure to ctx profile");
      Writer.writeContextual(*TopList, DC.TotalRootEntryCount);
    }
    Writer.endContextSection();
  }
  if (!SPR.FlatProfiles.empty()) {
    Writer.startFlatSection();
    for (const auto &[Guid, Counters] : SPR.FlatProfiles)
      Writer.writeFlat(Guid, Counters.data(), Counters.size());
    Writer.endFlatSection();
  }
  if (EC)
    return createStringError(EC, "failed to write output");
  return Error::success();
}
