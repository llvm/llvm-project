//===----- COFF_arm64.cpp - JIT linker implementation for COFF/arm64 ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// COFF/arm64 jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/COFF_arm64.h"
#include "COFFLinkGraphBuilder.h"
#include "SEHFrameSupport.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/ExecutionEngine/JITLink/aarch64.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Endian.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;

namespace {

enum EdgeKind_coff_arm64 : Edge::Kind {
  Pointer32 = aarch64::FirstPlatformRelocation,
  Pointer32NB,
  Branch26,
  PageBase_Rel21,
  Rel21,
  PageOffset_12L,
  Secrel,
  Secrel_Low12A,
  Secrel_High12A,
  Secrel_Low12L,
  Token,
  Sec,
  Pointer64,
  Branch19,
  Branch14,
  Rel32
};

class COFFJITLinker_arm64 : public JITLinker<COFFJITLinker_arm64> {
  friend class JITLinker<COFFJITLinker_arm64>;

public:
  COFFJITLinker_arm64(std::unique_ptr<JITLinkContext> Ctx,
                      std::unique_ptr<LinkGraph> G,
                      PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    return aarch64::applyFixup(G, B, E);
  }
};

class COFFLinkGraphBuilder_arm64 : public COFFLinkGraphBuilder {
private:
  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Processing relocations:\n");

    for (const auto &RelSect : sections())
      if (Error Err = COFFLinkGraphBuilder::forEachRelocation(
              RelSect, this, &COFFLinkGraphBuilder_arm64::addSingleRelocation))
        return Err;

    return Error::success();
  }

  Error addSingleRelocation(const object::RelocationRef &Rel,
                            const object::SectionRef &FixupSect,
                            Block &BlockToFix) {
    const object::coff_relocation *COFFRel = getObject().getCOFFRelocation(Rel);
    auto SymbolIt = Rel.getSymbol();
    if (SymbolIt == getObject().symbol_end()) {
      return make_error<StringError>(
          formatv("Invalid symbol index in relocation entry. "
                  "index: {0}, section: {1}",
                  COFFRel->SymbolTableIndex, FixupSect.getIndex()),
          inconvertibleErrorCode());
    }

    object::COFFSymbolRef COFFSymbol = getObject().getCOFFSymbol(*SymbolIt);
    COFFSymbolIndex SymIndex = getObject().getSymbolIndex(COFFSymbol);

    Symbol *GraphSymbol = getGraphSymbol(SymIndex);
    if (!GraphSymbol)
      return make_error<StringError>(
          formatv("Could not find symbol at given index, did you add it to "
                  "JITSymbolTable? index: {0}, section: {1}",
                  SymIndex, FixupSect.getIndex()),
          inconvertibleErrorCode());

    int64_t Addend = 0;
    orc::ExecutorAddr FixupAddress =
        orc::ExecutorAddr(FixupSect.getAddress()) + Rel.getOffset();
    Edge::OffsetT Offset = FixupAddress - BlockToFix.getAddress();

    Edge::Kind Kind = Edge::Invalid;
    const char *FixupPtr = BlockToFix.getContent().data() + Offset;

    switch (Rel.getType()) {
    case COFF::RelocationTypesARM64::IMAGE_REL_ARM64_ADDR32: {
      Kind = EdgeKind_coff_arm64::Pointer32;
      Addend = *reinterpret_cast<const support::little32_t *>(FixupPtr);
      break;
    }
    case COFF::RelocationTypesARM64::IMAGE_REL_ARM64_ADDR32NB: {
      Kind = EdgeKind_coff_arm64::Pointer32NB;
      Addend = *reinterpret_cast<const support::little32_t *>(FixupPtr);
      break;
    }
    case COFF::RelocationTypesARM64::IMAGE_REL_ARM64_BRANCH26: {
      Kind = EdgeKind_coff_arm64::Branch26;
      Addend = *reinterpret_cast<const support::little32_t *>(FixupPtr) >> 4;
      break;
    }
    case COFF::RelocationTypesARM64::IMAGE_REL_ARM64_PAGEBASE_REL21: {
      Kind = EdgeKind_coff_arm64::PageBase_Rel21;
      Addend = *reinterpret_cast<const support::little32_t *>(FixupPtr);
      break;
    }
    case COFF::RelocationTypesARM64::IMAGE_REL_ARM64_REL21: {
      Kind = EdgeKind_coff_arm64::Rel21;
      Addend = *reinterpret_cast<const support::little32_t *>(FixupPtr);
      break;
    }
    case COFF::RelocationTypesARM64::IMAGE_REL_ARM64_BRANCH19: {
      Kind = EdgeKind_coff_arm64::Branch19;
      Addend = *reinterpret_cast<const support::little32_t *>(FixupPtr) >> 11;
      break;
    }
    case COFF::RelocationTypesARM64::IMAGE_REL_ARM64_BRANCH14: {
      Kind = EdgeKind_coff_arm64::Branch14;
      Addend = *reinterpret_cast<const support::little32_t *>(FixupPtr) >> 16;
      break;
    }
    default: {
      return make_error<JITLinkError>("Unsupported arm64 relocation:" +
                                      formatv("{0:d}", Rel.getType()));
    }
    }

    Edge GE(Kind, Offset, *GraphSymbol, Addend);
    LLVM_DEBUG({
      dbgs() << "    ";
      printEdge(dbgs(), BlockToFix, GE, getCOFFARM64RelocationKindName(Kind));
      dbgs() << "\n";
    });

    BlockToFix.addEdge(std::move(GE));

    return Error::success();
  }

public:
  COFFLinkGraphBuilder_arm64(const object::COFFObjectFile &Obj, const Triple T,
                             const SubtargetFeatures Features)
      : COFFLinkGraphBuilder(Obj, std::move(T), std::move(Features),
                             getCOFFARM64RelocationKindName) {}
};

class COFFLinkGraphLowering_arm64 {
public:
  // Lowers COFF arm64 specific edges to generic arm64 edges.
  Error lowerCOFFRelocationEdges(LinkGraph &G, JITLinkContext &Ctx) {
    for (auto *B : G.blocks()) {
      for (auto &E : B->edges()) {
        switch (E.getKind()) {
        case EdgeKind_coff_arm64::Pointer32: {
          E.setKind(aarch64::Pointer32);
          break;
        }
        case EdgeKind_coff_arm64::Pointer32NB: {
          E.setKind(aarch64::Pointer32);
          break;
        }
        case EdgeKind_coff_arm64::Branch26: {
          E.setKind(aarch64::Branch26PCRel);
          break;
        }
        case EdgeKind_coff_arm64::PageBase_Rel21: {
          E.setKind(aarch64::Page21);
          break;
        }
        case EdgeKind_coff_arm64::Rel21: {
          // TODO
          break;
        }
        case EdgeKind_coff_arm64::PageOffset_12L: {
          E.setKind(aarch64::PageOffset12);
          break;
        }
        case EdgeKind_coff_arm64::Secrel: {
          // TODO
          break;
        }
        case EdgeKind_coff_arm64::Secrel_Low12A: {
          // TODO
          break;
        }
        case EdgeKind_coff_arm64::Secrel_High12A: {
          // TODO
          break;
        }
        case EdgeKind_coff_arm64::Secrel_Low12L: {
          // TODO
          break;
        }
        case EdgeKind_coff_arm64::Token: {
          // TODO
          break;
        }
        case EdgeKind_coff_arm64::Sec: {
          // TODO
          break;
        }
        case EdgeKind_coff_arm64::Pointer64: {
          E.setKind(aarch64::Pointer64);
          break;
        }
        case EdgeKind_coff_arm64::Branch19: {
          E.setKind(aarch64::CondBranch19PCRel);
          break;
        }
        case EdgeKind_coff_arm64::Branch14: {
          E.setKind(aarch64::TestAndBranch14PCRel);
          break;
        }
        case EdgeKind_coff_arm64::Rel32: {
          // TODO
          break;
        }
        default:
          break;
        }
      }
    }

    return Error::success();
  }
};

Error lowerEdges_COFF_arm64(LinkGraph &G, JITLinkContext *Ctx) {
  LLVM_DEBUG(dbgs() << "Lowering to generic COFF arm64 edges:\n");
  COFFLinkGraphLowering_arm64 GraphLowering;

  if (auto Err = GraphLowering.lowerCOFFRelocationEdges(G, *Ctx))
    return Err;

  return Error::success();
}
} // namespace

namespace llvm {
namespace jitlink {

/// Return the string name of the given COFF ARM64 edge kind.
const char *getCOFFARM64RelocationKindName(Edge::Kind R) {
  switch (R) {
  case Pointer32:
    return "Pointer32";
  case Pointer32NB:
    return "Pointer32NB";
  case Branch26:
    return "Branch26";
  case PageBase_Rel21:
    return "PageBase_Rel21";
  case Rel21:
    return "Rel21";
  case PageOffset_12L:
    return "PageOffset_12L";
  case Secrel:
    return "Secrel";
  case Secrel_Low12A:
    return "Secrel_Low12A";
  case Secrel_High12A:
    return "Secrel_High12A";
  case Secrel_Low12L:
    return "Secrel_Low12L";
  case Token:
    return "Token";
  case Sec:
    return "Section";
  case Pointer64:
    return "Pointer64";
  case Branch19:
    return "Branch19";
  case Branch14:
    return "Branch14";
  case Rel32:
    return "Rel32";
  default:
    return aarch64::getEdgeKindName(R);
  }
}

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromCOFFObject_arm64(MemoryBufferRef ObjectBuffer) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  auto COFFObj = object::ObjectFile::createCOFFObjectFile(ObjectBuffer);
  if (!COFFObj)
    return COFFObj.takeError();

  auto Features = (*COFFObj)->getFeatures();
  if (!Features)
    return Features.takeError();

  return COFFLinkGraphBuilder_arm64(**COFFObj, (*COFFObj)->makeTriple(),
                                    std::move(*Features))
      .buildGraph();
}

void link_COFF_arm64(std::unique_ptr<LinkGraph> G,
                     std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  const Triple &TT = G->getTargetTriple();
  if (Ctx->shouldAddDefaultTargetPasses(TT)) {
    // Add a mark-live pass.
    if (auto MarkLive = Ctx->getMarkLivePass(TT)) {
      Config.PrePrunePasses.push_back(std::move(MarkLive));
      Config.PrePrunePasses.push_back(SEHFrameKeepAlivePass(".pdata"));
    } else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);

    // Add COFF edge lowering passes.
    JITLinkContext *CtxPtr = Ctx.get();
    Config.PreFixupPasses.push_back(
        [CtxPtr](LinkGraph &G) { return lowerEdges_COFF_arm64(G, CtxPtr); });
  }

  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  COFFJITLinker_arm64::link(std::move(Ctx), std::move(G), std::move(Config));
}
} // namespace jitlink
} // namespace llvm
