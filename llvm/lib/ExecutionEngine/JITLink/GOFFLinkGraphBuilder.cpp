//===--- GOFFLinkGraphBuilder.cpp - GOFF LinkGraph Builder ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic GOFF LinkGraph building code.
//
//===----------------------------------------------------------------------===//

#include "GOFFLinkGraphBuilder.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/ExecutionEngine/JITLink/GOFF_systemz.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/systemz.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/MemoryFlags.h"
#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::object;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

GOFFLinkGraphBuilder::GOFFLinkGraphBuilder(
    const object::GOFFObjectFile &Obj,
    std::shared_ptr<orc::SymbolStringPool> SSP, Triple TT,
    SubtargetFeatures Features,
    LinkGraph::GetEdgeKindNameFunction GetEdgeKindName)
    : Obj(Obj),
      G(std::make_unique<LinkGraph>(
          std::string(Obj.getFileName()), std::move(SSP), std::move(TT),
          std::move(Features), std::move(GetEdgeKindName))) {}

Expected<std::unique_ptr<LinkGraph>> GOFFLinkGraphBuilder::buildGraph() {
  LLVM_DEBUG(dbgs() << "Building GOFFLinkGraph...\n");

  // Check to make sure the object is relocatable.
  if (!Obj.isRelocatableObject())
    return make_error<JITLinkError>("Object is not a relocatable COFF file");

  if (auto Err = processSections())
    return std::move(Err);
  if (auto Err = processSymbols())
    return std::move(Err);
  if (auto Err = processRelocations())
    return std::move(Err);

  return std::move(G);
}

Error GOFFLinkGraphBuilder::processSections() {
  LLVM_DEBUG(dbgs() << "Processing GOFF sections ...\n");

  for (const object::SectionRef Sec : Obj.sections()) {
    Expected<StringRef> NameOrErr = Sec.getName();
    if (!NameOrErr) {
      return NameOrErr.takeError();
    }

    // Skip empty sections.
    if (Sec.getSize() == 0)
      return Error::success();

    StringRef SectionName = *NameOrErr;
    DataRefImpl SecRawDataRef = Sec.getRawDataRefImpl();
    SmallString<16> UniqSectionName;
    if (Error Err = Obj.getSectionUniqueName(SecRawDataRef, UniqSectionName)) {
      return Err;
    }

    LLVM_DEBUG({
      dbgs() << "    section = " << SectionName
             << ", uniq = " << UniqSectionName << ", idx = " << Sec.getIndex()
             << ", size = " << format_hex_no_prefix(Sec.getSize(), 8)
             << ", vma = " << format_hex(Sec.getAddress(), 16) << "\n";
    });

    // Skip debug sections.
    if (Sec.isDebugSection())
      continue;

    // Get memory protection flags.
    orc::MemProt Prot = orc::MemProt::Read;
    if (Sec.isText())
      Prot |= orc::MemProt::Exec;
    else if (Sec.isData())
      Prot |= orc::MemProt::Write;

    // Get or create the section in the graph.
    auto *GraphSec = G->findSectionByName(UniqSectionName);
    assert(!GraphSec && "Should be named unique");
    GraphSec = &G->createSection(UniqSectionName, Prot);

    if (GraphSec->getMemProt() != Prot)
      return make_error<JITLinkError>("MemProt should match");

    uint32_t SecIndex = Sec.getIndex();
    if (SectionMap.contains(SecIndex))
      return make_error<JITLinkError>("Index already exists");

    Expected<StringRef> ContentsOrErr = Sec.getContents();
    if (!ContentsOrErr) {
      return ContentsOrErr.takeError();
    }

    // Create the content block in the graph.
    StringRef Contents = *ContentsOrErr;
    uint64_t SecAddress = Sec.getAddress();
    Block *B = &G->createContentBlock(*GraphSec, Contents,
                                      orc::ExecutorAddr(SecAddress),
                                      Sec.getAlignment().value(), 0);
    SectionMap[SecIndex] = {GraphSec, B, Sec};
  }

  return Error::success();
}

Error GOFFLinkGraphBuilder::processSymbols() {
  LLVM_DEBUG(dbgs() << "Processing GOFF symbols...\n");

  for (object::GOFFSymbolRef Sym : Obj.symbols()) {
    Expected<StringRef> NameOrErr = Sym.getName();
    if (!NameOrErr)
      return NameOrErr.takeError();

    StringRef Name = *NameOrErr;
    uint32_t SymEsdId = Sym.getRawDataRefImpl().d.a;
    LLVM_DEBUG(dbgs() << "  Processing symbol [" << SymEsdId << "] " << Name
                      << "\n");

    Expected<uint32_t> SymFlagsOrErr = Sym.getSymbolGOFFFlags();
    if (!SymFlagsOrErr)
      return SymFlagsOrErr.takeError();

    uint32_t Flags = *SymFlagsOrErr;
    if (Flags & object::SymbolRef::SF_Undefined) {
      LLVM_DEBUG(dbgs() << "      created external symbol\n");
      SymbolMap[SymEsdId] = &G->addExternalSymbol(
          Name, Sym.getSize(), Flags & object::SymbolRef::SF_Weak);
      continue;
    }

    auto SymbolTypeOrErr = Sym.getSymbolGOFFType();
    if (!SymbolTypeOrErr)
      return SymbolTypeOrErr.takeError();

    Expected<section_iterator> SectionOrErr = Sym.getSection();
    if (!SectionOrErr)
      return SectionOrErr.takeError();

    section_iterator SI = *SectionOrErr;
    if (SI == Obj.section_end())
      return make_error<JITLinkError>("Symbol section not found");

    Expected<uint64_t> OffsetOrErr = Sym.getAddress();
    if (!OffsetOrErr)
      return OffsetOrErr.takeError();

    uint32_t SecIndex = SI->getIndex();
    Block *B = SectionMap[SecIndex].Block;
    uint64_t Offset = *OffsetOrErr;
    Linkage L =
        (Flags & object::SymbolRef::SF_Weak) ? Linkage::Weak : Linkage::Strong;
    Scope S{Scope::Local};
    if (Flags & object::SymbolRef::SF_Hidden)
      S = Scope::Hidden;
    else if (Flags & object::SymbolRef::SF_Global)
      S = Scope::Default;
    SymbolRef::Type SymbolType = *SymbolTypeOrErr;
    bool IsCallable = (SymbolType == object::SymbolRef::ST_Function);

    LLVM_DEBUG(dbgs() << "      creating with linkage = " << getLinkageName(L)
                      << ", scope = " << getScopeName(S)
                      << ", B = " << format_hex(B->getAddress().getValue(), 16)
                      << (IsCallable ? " function" : " non-callable") << "\n");

    SymbolMap[SymEsdId] = &G->addDefinedSymbol(*B, Offset, Name, Sym.getSize(),
                                               L, S, IsCallable, true);
  }

  return Error::success();
}

static systemz::EdgeKind_systemz getRelEdgeKind(uint64_t RelType) {
  GOFF::RLDReferenceType RldRefType = getRLDReferenceType(RelType);
  GOFF::RLDAction RldAct = getRLDAction(RelType);
  GOFF::RLDFetchStore RldFetch = getRLDFetchStore(RelType);
  uint8_t RldLength = getRLDTargetLength(RelType);
  uint8_t RldBitLength = getRLDBitLength(RelType);
  uint8_t RldBitWidth = 8 * RldLength + RldBitLength;

  switch (RldRefType) {
  case GOFF::RLD_RT_RAddress:
    switch (RldBitWidth) {
    case 64:
      if (RldFetch == GOFF::RLD_FS_Fetch)
        return (RldAct == GOFF::RLD_ACT_Add ? systemz::Pointer64Add
                                            : systemz::Pointer64Sub);
      else
        return systemz::Pointer64;
      break;
    case 32:
      if (RldFetch == GOFF::RLD_FS_Fetch)
        return (RldAct == GOFF::RLD_ACT_Add ? systemz::Pointer32Add
                                            : systemz::Pointer32Sub);
      else
        return systemz::Pointer32;
      break;
    default:
      llvm_unreachable("Unsuppoted rld reference type");
    }
    break;
  default:
    llvm_unreachable("Unsuppoted rld reference type");
  }
}

Error GOFFLinkGraphBuilder::processRelocations() {
  LLVM_DEBUG(dbgs() << "Processing GOFF relocations...\n");

  for (const object::SectionRef Sec : Obj.sections()) {
    uint32_t SecIndex = Sec.getIndex();
    auto SectionName = Sec.getName();
    if (!SectionName)
      return SectionName.takeError();

    LLVM_DEBUG(dbgs() << " Relocations for section " << *SectionName << "\n");

    for (object::RelocationRef Relocation : Sec.relocations()) {
      object::SymbolRef Sym = *Relocation.getSymbol();
      auto TargetNameOrErr = Sym.getName();
      if (!TargetNameOrErr) {
        return TargetNameOrErr.takeError();
      }

      SmallString<16> RelTypeName;
      Relocation.getTypeName(RelTypeName);
      uint64_t RelType = Relocation.getType();
      systemz::EdgeKind_systemz EK = getRelEdgeKind(RelType);
      jitlink::Block *B = SectionMap[SecIndex].Block;
      uint32_t TargetBlockOffset = Sec.getAddress() + Relocation.getOffset() -
                                   B->getAddress().getValue();
      uint32_t REsdId = Sym.getRawDataRefImpl().d.a;
      jitlink::Symbol *S = SymbolMap[REsdId];

      LLVM_DEBUG({
        dbgs() << "    reloffset = " << format_hex(Relocation.getOffset(), 16)
               << " typename =  " << RelTypeName << " idx =  " << REsdId
               << " block = (" << B << ", "
               << format_hex(B->getAddress().getValue(), 16) << ")"
               << " targetname: " << *TargetNameOrErr << "\n";
      });

      B->addEdge(EK, TargetBlockOffset, *S, 0);
    }
  }
  return Error::success();
}

} // namespace jitlink
} // namespace llvm
