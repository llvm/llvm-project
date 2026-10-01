//===----------- GOFF_systemz.cpp - JIT linker function for GOFF ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// JIT-Link functions for GOFF/systemz.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/GOFF_systemz.h"
#include "GOFFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"
#include "llvm/ExecutionEngine/JITLink/GOFF.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/systemz.h"
#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <system_error>

using namespace llvm;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

class GOFFJITLinker_systemz : public JITLinker<GOFFJITLinker_systemz> {
  using JITLinkerBase = JITLinker<GOFFJITLinker_systemz>;
  friend JITLinkerBase;

public:
  GOFFJITLinker_systemz(std::unique_ptr<JITLinkContext> Ctx,
                        std::unique_ptr<LinkGraph> G,
                        PassConfiguration PassConfig)
      : JITLinkerBase(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    if (auto Err = systemz::applyGOFFFixup(G, B, E)) {
      return make_error<StringError>("Unsupported goff relocation type",
                                     std::error_code());
    }
    return Error::success();
  }
};

class GOFFLinkGraphBuilder_systemz : public GOFFLinkGraphBuilder {
private:
  Error processRelocations() override;

public:
  GOFFLinkGraphBuilder_systemz(const object::GOFFObjectFile &Obj,
                               std::shared_ptr<orc::SymbolStringPool> SSP,
                               const Triple T, const SubtargetFeatures Features)
      : GOFFLinkGraphBuilder(Obj, std::move(SSP), std::move(T),
                             std::move(Features), systemz::getEdgeKindName) {}
};

Expected<std::unique_ptr<LinkGraph>> createLinkGraphFromGOFFObject_systemz(
    MemoryBufferRef ObjectBuffer, std::shared_ptr<orc::SymbolStringPool> SSP) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  file_magic Magic = identify_magic(ObjectBuffer.getBuffer());
  if (Magic != file_magic::goff_object)
    return make_error<JITLinkError>("Invalid GOFF Header");

  auto GOFFObj = object::ObjectFile::createObjectFile(ObjectBuffer);
  if (!GOFFObj)
    return GOFFObj.takeError();
  assert((*GOFFObj)->isGOFF() && "Expects an GOFF Object");
  assert((*GOFFObj)->getArch() == Triple::systemz && "Only support systemz");

  auto Features = (*GOFFObj)->getFeatures();
  if (!Features)
    return Features.takeError();
  LLVM_DEBUG({
    dbgs() << " Features: ";
    (*Features).print(dbgs());
  });

  // Set the flag to preserve GOFF ED symbols for creating JITLink symbols.
  cast<object::GOFFObjectFile>(**GOFFObj).setSkipEDSymbols(false);

  return GOFFLinkGraphBuilder_systemz(cast<object::GOFFObjectFile>(**GOFFObj),
                                      std::move(SSP), (*GOFFObj)->makeTriple(),
                                      std::move(*Features))
      .buildGraph();
}

void link_GOFF_systemz(std::unique_ptr<LinkGraph> G,
                       std::unique_ptr<JITLinkContext> Ctx) {

  PassConfiguration PassCfg;
  if (auto Err = Ctx->modifyPassConfig(*G, PassCfg))
    return Ctx->notifyFailed(std::move(Err));

  if (G->getTargetTriple().getArch() != Triple::systemz)
    return Ctx->notifyFailed(make_error<JITLinkError>(
        "Unsupported target machine architecture in GOFF link graph " +
        G->getName()));

  GOFFJITLinker_systemz::link(std::move(Ctx), std::move(G), std::move(PassCfg));
  return;
}

static systemz::EdgeKind_systemz getRelEdgeKind(uint64_t RelType) {
  GOFF::RLDReferenceType RldRefType = object::getRLDReferenceType(RelType);
  GOFF::RLDAction RldAct = object::getRLDAction(RelType);
  GOFF::RLDFetchStore RldFetch = object::getRLDFetchStore(RelType);
  uint8_t RldLength = object::getRLDTargetLength(RelType);
  uint8_t RldBitLength = object::getRLDBitLength(RelType);
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

Error GOFFLinkGraphBuilder_systemz::processRelocations() {
  LLVM_DEBUG(dbgs() << "Processing GOFF relocations...\n");

  for (const object::SectionRef Sec : sections()) {
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
      jitlink::Block *B = getGraphBlock(SecIndex);
      assert(B && "Block not found");

      uint32_t TargetBlockOffset = Sec.getAddress() + Relocation.getOffset() -
                                   B->getAddress().getValue();
      uint32_t REsdId = Sym.getRawDataRefImpl().d.a;
      jitlink::Symbol *S = getGraphSymbol(REsdId);
      assert(S && "Symbol not found");

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
