//===--- ELF_loongarch.cpp - JIT linker implementation for ELF/loongarch --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/loongarch jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_loongarch.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/DWARFRecordSectionSplitter.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/loongarch.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"

#include "EHFrameSupportImpl.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::loongarch;

namespace {

class ELFJITLinker_loongarch : public JITLinker<ELFJITLinker_loongarch> {
  friend class JITLinker<ELFJITLinker_loongarch>;

public:
  ELFJITLinker_loongarch(std::unique_ptr<JITLinkContext> Ctx,
                         std::unique_ptr<LinkGraph> G,
                         PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {
    JITLinkerBase::getPassConfig().PostAllocationPasses.push_back(
        [this](LinkGraph &G) { return gatherLoongArchPCAddHi20(G); });
  }

private:
  DenseMap<std::pair<const Block *, orc::ExecutorAddrDiff>, const Edge *>
      RelPCAddHi20Map;

  Error gatherLoongArchPCAddHi20(LinkGraph &G) {
    for (Block *B : G.blocks())
      for (Edge &E : B->edges())
        if (E.getKind() == PCAddHi20)
          RelPCAddHi20Map[{B, E.getOffset()}] = &E;

    return Error::success();
  }

  Expected<const Edge &> getLoongArchPCAddHi20(const Edge &E) const {
    using namespace loongarch;
    assert((E.getKind() == PCAddLo12) &&
           "Can only have high relocation for PCAddLo12");

    const Symbol &Sym = E.getTarget();
    const Block &B = Sym.getBlock();
    orc::ExecutorAddrDiff Offset = Sym.getOffset() + E.getAddend();

    auto It = RelPCAddHi20Map.find({&B, Offset});
    if (It != RelPCAddHi20Map.end())
      return *It->second;

    return make_error<JITLinkError>("No PCAddHi20 relocation type be found "
                                    "for PCAddLo12 relocation type");
  }

  /// Apply fixup expression for edge to block content.
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    using namespace support;

    char *BlockWorkingMem = B.getAlreadyMutableContent().data();
    char *FixupPtr = BlockWorkingMem + E.getOffset();
    uint64_t FixupAddress = (B.getAddress() + E.getOffset()).getValue();
    uint64_t TargetAddress = E.getTarget().getAddress().getValue();
    int64_t Addend = E.getAddend();

    switch (E.getKind()) {
    case Pointer64:
      *(ulittle64_t *)FixupPtr = TargetAddress + Addend;
      break;
    case Pointer32: {
      uint64_t Value = TargetAddress + Addend;
      if (Value > std::numeric_limits<uint32_t>::max())
        return makeTargetOutOfRangeError(G, B, E);
      *(ulittle32_t *)FixupPtr = Value;
      break;
    }
    case Branch16PCRel: {
      int64_t Value = TargetAddress - FixupAddress + Addend;

      if (!isInt<18>(Value))
        return makeTargetOutOfRangeError(G, B, E);

      if (!isShiftedInt<16, 2>(Value))
        return makeAlignmentError(orc::ExecutorAddr(FixupAddress), Value, 4, E);

      uint32_t RawInstr = *(little32_t *)FixupPtr;
      uint32_t Imm = static_cast<uint32_t>(Value >> 2);
      uint32_t Imm15_0 = extractBits(Imm, /*Hi=*/15, /*Lo=*/0) << 10;
      *(little32_t *)FixupPtr = RawInstr | Imm15_0;
      break;
    }
    case Branch21PCRel: {
      int64_t Value = TargetAddress - FixupAddress + Addend;

      if (!isInt<23>(Value))
        return makeTargetOutOfRangeError(G, B, E);

      if (!isShiftedInt<21, 2>(Value))
        return makeAlignmentError(orc::ExecutorAddr(FixupAddress), Value, 4, E);

      uint32_t RawInstr = *(little32_t *)FixupPtr;
      uint32_t Imm = static_cast<uint32_t>(Value >> 2);
      uint32_t Imm15_0 = extractBits(Imm, /*Hi=*/15, /*Lo=*/0) << 10;
      uint32_t Imm20_16 = extractBits(Imm, /*Hi=*/20, /*Lo=*/16);
      *(little32_t *)FixupPtr = RawInstr | Imm15_0 | Imm20_16;
      break;
    }
    case Branch26PCRel: {
      int64_t Value = TargetAddress - FixupAddress + Addend;

      if (!isInt<28>(Value))
        return makeTargetOutOfRangeError(G, B, E);

      if (!isShiftedInt<26, 2>(Value))
        return makeAlignmentError(orc::ExecutorAddr(FixupAddress), Value, 4, E);

      uint32_t RawInstr = *(little32_t *)FixupPtr;
      uint32_t Imm = static_cast<uint32_t>(Value >> 2);
      uint32_t Imm15_0 = extractBits(Imm, /*Hi=*/15, /*Lo=*/0) << 10;
      uint32_t Imm25_16 = extractBits(Imm, /*Hi=*/25, /*Lo=*/16);
      *(little32_t *)FixupPtr = RawInstr | Imm15_0 | Imm25_16;
      break;
    }
    case Delta32: {
      int64_t Value = TargetAddress - FixupAddress + Addend;

      if (!isInt<32>(Value))
        return makeTargetOutOfRangeError(G, B, E);
      *(little32_t *)FixupPtr = Value;
      break;
    }
    case NegDelta32: {
      int64_t Value = FixupAddress - TargetAddress + Addend;
      if (!isInt<32>(Value))
        return makeTargetOutOfRangeError(G, B, E);
      *(little32_t *)FixupPtr = Value;
      break;
    }
    case Delta64:
      *(little64_t *)FixupPtr = TargetAddress - FixupAddress + Addend;
      break;
    case Page20: {
      uint64_t Target = TargetAddress + Addend;
      uint64_t TargetPage =
          (Target + (Target & 0x800)) & ~static_cast<uint64_t>(0xfff);
      uint64_t PCPage = FixupAddress & ~static_cast<uint64_t>(0xfff);

      int64_t PageDelta = TargetPage - PCPage;
      if (!isInt<32>(PageDelta))
        return makeTargetOutOfRangeError(G, B, E);

      uint32_t RawInstr = *(little32_t *)FixupPtr;
      uint32_t Imm31_12 = extractBits(PageDelta, /*Hi=*/31, /*Lo=*/12) << 5;
      *(little32_t *)FixupPtr = RawInstr | Imm31_12;
      break;
    }
    case PageOffset12: {
      uint64_t TargetOffset = (TargetAddress + Addend) & 0xfff;

      uint32_t RawInstr = *(ulittle32_t *)FixupPtr;
      uint32_t Imm11_0 = TargetOffset << 10;
      *(ulittle32_t *)FixupPtr = RawInstr | Imm11_0;
      break;
    }
    case PCAddHi20: {
      uint64_t Target = TargetAddress + Addend;
      int64_t Delta = Target - FixupAddress + 0x800;

      if (!isInt<32>(Delta))
        return makeTargetOutOfRangeError(G, B, E);

      uint32_t RawInstr = *(little32_t *)FixupPtr;
      uint32_t Imm31_12 = extractBits(Delta, /*Hi=*/31, /*Lo=*/12) << 5;
      *(little32_t *)FixupPtr = RawInstr | Imm31_12;
      break;
    }
    case PCAddLo12: {
      auto RelPCAddHi20 = getLoongArchPCAddHi20(E);
      if (!RelPCAddHi20)
        return RelPCAddHi20.takeError();
      int64_t Delta =
          (RelPCAddHi20->getTarget().getAddress() + RelPCAddHi20->getAddend()) -
          (E.getTarget().getAddress() + E.getAddend());

      uint32_t RawInstr = *(ulittle32_t *)FixupPtr;
      uint32_t Imm11_0 = extractBits(Delta, /*Hi=*/11, /*Lo=*/0) << 10;
      *(ulittle32_t *)FixupPtr = RawInstr | Imm11_0;
      break;
    }
    case Call30PCRel: {
      int64_t Value = TargetAddress - FixupAddress + Addend;

      if (Value != llvm::SignExtend64(Value, 32))
        return makeTargetOutOfRangeError(G, B, E);

      if (!isShiftedInt<30, 2>(Value))
        return makeAlignmentError(orc::ExecutorAddr(FixupAddress), Value, 4, E);

      uint32_t Pcaddu12i = *(little32_t *)FixupPtr;
      uint32_t Hi20 = extractBits(Value, /*Hi=*/31, /*Lo=*/12) << 5;
      *(little32_t *)FixupPtr = Pcaddu12i | Hi20;
      uint32_t Jirl = *(little32_t *)(FixupPtr + 4);
      uint32_t Lo10 = extractBits(Value, /*Hi=*/11, /*Lo=*/2) << 10;
      *(little32_t *)(FixupPtr + 4) = Jirl | Lo10;
      break;
    }
    case Call36PCRel: {
      int64_t Value = TargetAddress - FixupAddress + Addend;

      if ((Value + 0x20000) != llvm::SignExtend64(Value + 0x20000, 38))
        return makeTargetOutOfRangeError(G, B, E);

      if (!isShiftedInt<36, 2>(Value))
        return makeAlignmentError(orc::ExecutorAddr(FixupAddress), Value, 4, E);

      uint32_t Pcaddu18i = *(little32_t *)FixupPtr;
      uint32_t Hi20 = extractBits(Value + (1 << 17), /*Hi=*/37, /*Lo=*/18) << 5;
      *(little32_t *)FixupPtr = Pcaddu18i | Hi20;
      uint32_t Jirl = *(little32_t *)(FixupPtr + 4);
      uint32_t Lo16 = extractBits(Value, /*Hi=*/17, /*Lo=*/2) << 10;
      *(little32_t *)(FixupPtr + 4) = Jirl | Lo16;
      break;
    }
    case Add6: {
      int64_t Value = *(reinterpret_cast<const int8_t *>(FixupPtr));
      Value += ((TargetAddress + Addend) & 0x3f);
      *FixupPtr = (*FixupPtr & 0xc0) | (static_cast<int8_t>(Value) & 0x3f);
      break;
    }
    case Add8: {
      int64_t Value = TargetAddress +
                      *(reinterpret_cast<const int8_t *>(FixupPtr)) + Addend;
      *FixupPtr = static_cast<int8_t>(Value);
      break;
    }
    case Add16: {
      int64_t Value =
          TargetAddress + support::endian::read16le(FixupPtr) + Addend;
      *(little16_t *)FixupPtr = static_cast<int16_t>(Value);
      break;
    }
    case Add32: {
      int64_t Value =
          TargetAddress + support::endian::read32le(FixupPtr) + Addend;
      *(little32_t *)FixupPtr = static_cast<int32_t>(Value);
      break;
    }
    case Add64: {
      int64_t Value =
          TargetAddress + support::endian::read64le(FixupPtr) + Addend;
      *(little64_t *)FixupPtr = static_cast<int64_t>(Value);
      break;
    }
    case AddUleb128: {
      const uint32_t Maxcount = 1 + 64 / 7;
      uint32_t Count;
      const char *Error = nullptr;
      uint64_t Orig =
          decodeULEB128((reinterpret_cast<const uint8_t *>(FixupPtr)), &Count,
                        nullptr, &Error);

      if (Count > Maxcount || (Count == Maxcount && Error))
        return make_error<JITLinkError>(
            "0x" + llvm::utohexstr(orc::ExecutorAddr(FixupAddress).getValue()) +
            ": extra space for uleb128");

      uint64_t Mask = Count < Maxcount ? (1ULL << 7 * Count) - 1 : -1ULL;
      encodeULEB128((Orig + TargetAddress + Addend) & Mask,
                    (reinterpret_cast<uint8_t *>(FixupPtr)), Count);
      break;
    }
    case Sub6: {
      int64_t Value = *(reinterpret_cast<const int8_t *>(FixupPtr));
      Value -= ((TargetAddress + Addend) & 0x3f);
      *FixupPtr = (*FixupPtr & 0xc0) | (static_cast<int8_t>(Value) & 0x3f);
      break;
    }
    case Sub8: {
      int64_t Value = *(reinterpret_cast<const int8_t *>(FixupPtr)) -
                      TargetAddress - Addend;
      *FixupPtr = static_cast<int8_t>(Value);
      break;
    }
    case Sub16: {
      int64_t Value =
          support::endian::read16le(FixupPtr) - TargetAddress - Addend;
      *(little16_t *)FixupPtr = static_cast<int16_t>(Value);
      break;
    }
    case Sub32: {
      int64_t Value =
          support::endian::read32le(FixupPtr) - TargetAddress - Addend;
      *(little32_t *)FixupPtr = static_cast<int32_t>(Value);
      break;
    }
    case Sub64: {
      int64_t Value =
          support::endian::read64le(FixupPtr) - TargetAddress - Addend;
      *(little64_t *)FixupPtr = static_cast<int64_t>(Value);
      break;
    }
    case SubUleb128: {
      const uint32_t Maxcount = 1 + 64 / 7;
      uint32_t Count;
      const char *Error = nullptr;
      uint64_t Orig =
          decodeULEB128((reinterpret_cast<const uint8_t *>(FixupPtr)), &Count,
                        nullptr, &Error);

      if (Count > Maxcount || (Count == Maxcount && Error))
        return make_error<JITLinkError>(
            "0x" + llvm::utohexstr(orc::ExecutorAddr(FixupAddress).getValue()) +
            ": extra space for uleb128");

      uint64_t Mask = Count < Maxcount ? (1ULL << 7 * Count) - 1 : -1ULL;
      encodeULEB128((Orig - TargetAddress - Addend) & Mask,
                    (reinterpret_cast<uint8_t *>(FixupPtr)), Count);
      break;
    }
    case AlignRelaxable:
      // Ignore when the relaxation pass did not run
      break;
    default:
      return make_error<JITLinkError>(
          "In graph " + G.getName() + ", section " + B.getSection().getName() +
          " unsupported edge kind " + getEdgeKindName(E.getKind()));
    }

    return Error::success();
  }
};

namespace {

struct SymbolAnchor {
  uint64_t Offset;
  Symbol *Sym;
  bool End; // true for the anchor of getOffset() + getSize()
};

struct BlockRelaxAux {
  // This records symbol start and end offsets which will be adjusted according
  // to the nearest RelocDeltas element.
  SmallVector<SymbolAnchor, 0> Anchors;
  // All edges that either 1) are R_LARCH_ALIGN or 2) have a R_LARCH_RELAX edge
  // at the same offset.
  SmallVector<Edge *, 0> RelaxEdges;
  // For RelaxEdges[I], the actual offset is RelaxEdges[I]->getOffset() - (I ?
  // RelocDeltas[I - 1] : 0).
  SmallVector<uint32_t, 0> RelocDeltas;
  // For RelaxEdges[I], the actual type is EdgeKinds[I].
  SmallVector<Edge::Kind, 0> EdgeKinds;
  // List of rewritten instructions. Contains one raw encoded instruction per
  // element in EdgeKinds that isn't Invalid or R_LARCH_ALIGN.
  SmallVector<uint32_t, 0> Writes;
};

struct RelaxAux {
  DenseMap<Block *, BlockRelaxAux> Blocks;
};

} // namespace

static bool shouldRelax(const Section &S) {
  return (S.getMemProt() & orc::MemProt::Exec) != orc::MemProt::None;
}

static bool isRelaxable(const Edge &E) {
  switch (E.getKind()) {
  default:
    return false;
  case AlignRelaxable:
    return true;
  }
}

static RelaxAux initRelaxAux(LinkGraph &G) {
  RelaxAux Aux;
  for (auto &S : G.sections()) {
    if (!shouldRelax(S))
      continue;
    for (auto *B : S.blocks()) {
      auto BlockEmplaceResult = Aux.Blocks.try_emplace(B);
      assert(BlockEmplaceResult.second && "Block encountered twice");
      auto &BlockAux = BlockEmplaceResult.first->second;

      for (auto &E : B->edges())
        if (isRelaxable(E))
          BlockAux.RelaxEdges.push_back(&E);

      if (BlockAux.RelaxEdges.empty()) {
        Aux.Blocks.erase(BlockEmplaceResult.first);
        continue;
      }

      const auto NumEdges = BlockAux.RelaxEdges.size();
      BlockAux.RelocDeltas.resize(NumEdges, 0);
      BlockAux.EdgeKinds.resize_for_overwrite(NumEdges);

      // Store anchors (offset and offset+size) for symbols.
      for (auto *Sym : S.symbols()) {
        if (!Sym->isDefined() || &Sym->getBlock() != B)
          continue;

        BlockAux.Anchors.push_back({Sym->getOffset(), Sym, false});
        BlockAux.Anchors.push_back(
            {Sym->getOffset() + Sym->getSize(), Sym, true});
      }
    }
  }

  // Sort anchors by offset so that we can find the closest relocation
  // efficiently. For a zero size symbol, ensure that its start anchor precedes
  // its end anchor. For two symbols with anchors at the same offset, their
  // order does not matter.
  for (auto &BlockAuxIter : Aux.Blocks) {
    llvm::sort(BlockAuxIter.second.Anchors, [](auto &A, auto &B) {
      return std::make_pair(A.Offset, A.End) < std::make_pair(B.Offset, B.End);
    });
  }

  return Aux;
}

static void relaxAlign(orc::ExecutorAddr Loc, const Edge &E, uint32_t &Remove,
                       Edge::Kind &NewEdgeKind) {
  const uint64_t Addend =
      !E.getTarget().isDefined() ? Log2_64(E.getAddend()) + 1 : E.getAddend();
  const uint64_t AllBytes = (1ULL << (Addend & 0xff)) - 4;
  const uint64_t Align = 1ULL << (Addend & 0xff);
  const uint64_t MaxBytes = Addend >> 8;
  const uint64_t Off = Loc.getValue() & (Align - 1);
  const uint64_t CurBytes = Off == 0 ? 0 : Align - Off;
  // All bytes beyond the alignment boundary should be removed.
  // If emit bytes more than max bytes to emit, remove all.
  if (MaxBytes != 0 && CurBytes > MaxBytes)
    Remove = AllBytes;
  else
    Remove = AllBytes - CurBytes;

  assert(static_cast<int32_t>(Remove) >= 0 &&
         "R_LARCH_ALIGN needs expanding the content");
  NewEdgeKind = AlignRelaxable;
}

static bool relaxBlock(LinkGraph &G, Block &Block, BlockRelaxAux &Aux) {
  const auto BlockAddr = Block.getAddress();
  bool Changed = false;
  ArrayRef<SymbolAnchor> SA = ArrayRef(Aux.Anchors);
  uint32_t Delta = 0;

  Aux.EdgeKinds.assign(Aux.EdgeKinds.size(), Edge::Invalid);
  Aux.Writes.clear();

  for (auto [I, E] : llvm::enumerate(Aux.RelaxEdges)) {
    const auto Loc = BlockAddr + E->getOffset() - Delta;
    auto &Cur = Aux.RelocDeltas[I];
    uint32_t Remove = 0;
    switch (E->getKind()) {
    case AlignRelaxable:
      relaxAlign(Loc, *E, Remove, Aux.EdgeKinds[I]);
      break;
    default:
      llvm_unreachable("Unexpected relaxable edge kind");
    }

    // For all anchors whose offsets are <= E->getOffset(), they are preceded by
    // the previous relocation whose RelocDeltas value equals Delta.
    // Decrease their offset and update their size.
    for (; SA.size() && SA[0].Offset <= E->getOffset(); SA = SA.slice(1)) {
      if (SA[0].End)
        SA[0].Sym->setSize(SA[0].Offset - Delta - SA[0].Sym->getOffset());
      else
        SA[0].Sym->setOffset(SA[0].Offset - Delta);
    }

    Delta += Remove;
    if (Delta != Cur) {
      Cur = Delta;
      Changed = true;
    }
  }

  for (const SymbolAnchor &A : SA) {
    if (A.End)
      A.Sym->setSize(A.Offset - Delta - A.Sym->getOffset());
    else
      A.Sym->setOffset(A.Offset - Delta);
  }

  return Changed;
}

static bool relaxOnce(LinkGraph &G, RelaxAux &Aux) {
  bool Changed = false;

  for (auto &[B, BlockAux] : Aux.Blocks)
    Changed |= relaxBlock(G, *B, BlockAux);

  return Changed;
}

static void finalizeBlockRelax(LinkGraph &G, Block &Block, BlockRelaxAux &Aux) {
  auto Contents = Block.getAlreadyMutableContent();
  auto *Dest = Contents.data();
  uint32_t Offset = 0;
  uint32_t Delta = 0;

  // Update section content: remove NOPs for R_LARCH_ALIGN and rewrite
  // instructions for relaxed relocations.
  for (auto [I, E] : llvm::enumerate(Aux.RelaxEdges)) {
    uint32_t Remove = Aux.RelocDeltas[I] - Delta;
    Delta = Aux.RelocDeltas[I];
    if (Remove == 0 && Aux.EdgeKinds[I] == Edge::Invalid)
      continue;

    // Copy from last location to the current relocated location.
    const auto Size = E->getOffset() - Offset;
    std::memmove(Dest, Contents.data() + Offset, Size);
    Dest += Size;
    Offset = E->getOffset() + Remove;
  }

  std::memmove(Dest, Contents.data() + Offset, Contents.size() - Offset);

  // Fixup edge offsets and kinds.
  Delta = 0;
  size_t I = 0;
  for (auto &E : Block.edges()) {
    E.setOffset(E.getOffset() - Delta);

    if (I < Aux.RelaxEdges.size() && Aux.RelaxEdges[I] == &E) {
      if (Aux.EdgeKinds[I] != Edge::Invalid)
        E.setKind(Aux.EdgeKinds[I]);

      Delta = Aux.RelocDeltas[I];
      ++I;
    }
  }

  // Remove AlignRelaxable edges: all other relaxable edges got modified and
  // will be used later while linking. Alignment is entirely handled here so we
  // don't need these edges anymore.
  for (auto IE = Block.edges().begin(); IE != Block.edges().end();) {
    if (IE->getKind() == AlignRelaxable)
      IE = Block.removeEdge(IE);
    else
      ++IE;
  }
}

static void finalizeRelax(LinkGraph &G, RelaxAux &Aux) {
  for (auto &[B, BlockAux] : Aux.Blocks)
    finalizeBlockRelax(G, *B, BlockAux);
}

static Error relax(LinkGraph &G) {
  auto Aux = initRelaxAux(G);
  while (relaxOnce(G, Aux)) {
  }
  finalizeRelax(G, Aux);
  return Error::success();
}

template <typename ELFT>
class ELFLinkGraphBuilder_loongarch : public ELFLinkGraphBuilder<ELFT> {
private:
  static Expected<loongarch::EdgeKind_loongarch>
  getRelocationKind(const uint32_t Type) {
    using namespace loongarch;
    switch (Type) {
    case ELF::R_LARCH_64:
      return Pointer64;
    case ELF::R_LARCH_32:
      return Pointer32;
    case ELF::R_LARCH_32_PCREL:
      return Delta32;
    case ELF::R_LARCH_B16:
      return Branch16PCRel;
    case ELF::R_LARCH_B21:
      return Branch21PCRel;
    case ELF::R_LARCH_B26:
      return Branch26PCRel;
    case ELF::R_LARCH_PCALA_HI20:
      return Page20;
    case ELF::R_LARCH_PCALA_LO12:
      return PageOffset12;
    case ELF::R_LARCH_GOT_PC_HI20:
      return RequestGOTAndTransformToPage20;
    case ELF::R_LARCH_GOT_PC_LO12:
      return RequestGOTAndTransformToPageOffset12;
    case ELF::R_LARCH_CALL30:
      return Call30PCRel;
    case ELF::R_LARCH_CALL36:
      return Call36PCRel;
    case ELF::R_LARCH_ADD6:
      return Add6;
    case ELF::R_LARCH_ADD8:
      return Add8;
    case ELF::R_LARCH_ADD16:
      return Add16;
    case ELF::R_LARCH_ADD32:
      return Add32;
    case ELF::R_LARCH_ADD64:
      return Add64;
    case ELF::R_LARCH_ADD_ULEB128:
      return AddUleb128;
    case ELF::R_LARCH_SUB6:
      return Sub6;
    case ELF::R_LARCH_SUB8:
      return Sub8;
    case ELF::R_LARCH_SUB16:
      return Sub16;
    case ELF::R_LARCH_SUB32:
      return Sub32;
    case ELF::R_LARCH_SUB64:
      return Sub64;
    case ELF::R_LARCH_SUB_ULEB128:
      return SubUleb128;
    case ELF::R_LARCH_ALIGN:
      return AlignRelaxable;
    case ELF::R_LARCH_PCADD_HI20:
      return PCAddHi20;
    case ELF::R_LARCH_PCADD_LO12:
    case ELF::R_LARCH_GOT_PCADD_LO12:
      return PCAddLo12;
    case ELF::R_LARCH_GOT_PCADD_HI20:
      return RequestGOTAndTransformToPCAddHi20;
    }

    return make_error<JITLinkError>(
        "Unsupported loongarch relocation:" + formatv("{0:d}: ", Type) +
        object::getELFRelocationTypeName(ELF::EM_LOONGARCH, Type));
  }

  EdgeKind_loongarch getRelaxableRelocationKind(EdgeKind_loongarch Kind) {
    // TODO: Implement more. Just ignore all relaxations now.
    return Kind;
  }

  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Processing relocations:\n");

    using Base = ELFLinkGraphBuilder<ELFT>;
    using Self = ELFLinkGraphBuilder_loongarch<ELFT>;
    for (const auto &RelSect : Base::Sections)
      if (Error Err = Base::forEachRelaRelocation(RelSect, this,
                                                  &Self::addSingleRelocation))
        return Err;

    return Error::success();
  }

  Error addSingleRelocation(const typename ELFT::Rela &Rel,
                            const typename ELFT::Shdr &FixupSect,
                            Block &BlockToFix) {
    using Base = ELFLinkGraphBuilder<ELFT>;

    uint32_t Type = Rel.getType(false);
    int64_t Addend = Rel.r_addend;

    // ignore
    if (Type == ELF::R_LARCH_MARK_LA)
      return Error::success();

    if (Type == ELF::R_LARCH_RELAX) {
      if (BlockToFix.edges_empty())
        return make_error<StringError>(
            "R_LARCH_RELAX without preceding relocation",
            inconvertibleErrorCode());

      auto &PrevEdge = *std::prev(BlockToFix.edges().end());
      auto Kind = static_cast<EdgeKind_loongarch>(PrevEdge.getKind());
      PrevEdge.setKind(getRelaxableRelocationKind(Kind));
      return Error::success();
    }

    Expected<loongarch::EdgeKind_loongarch> Kind = getRelocationKind(Type);
    if (!Kind)
      return Kind.takeError();

    uint32_t SymbolIndex = Rel.getSymbol(false);
    auto ObjSymbol = Base::Obj.getRelocationSymbol(Rel, Base::SymTabSec);
    if (!ObjSymbol)
      return ObjSymbol.takeError();

    Symbol *GraphSymbol = Base::getGraphSymbol(SymbolIndex);
    if (!GraphSymbol)
      return make_error<StringError>(
          formatv("Could not find symbol at given index, did you add it to "
                  "JITSymbolTable? index: {0}, shndx: {1} Size of table: {2}",
                  SymbolIndex, (*ObjSymbol)->st_shndx,
                  Base::GraphSymbols.size()),
          inconvertibleErrorCode());

    auto FixupAddress = orc::ExecutorAddr(FixupSect.sh_addr) + Rel.r_offset;
    Edge::OffsetT Offset = FixupAddress - BlockToFix.getAddress();
    Edge GE(*Kind, Offset, *GraphSymbol, Addend);
    LLVM_DEBUG({
      dbgs() << "    ";
      printEdge(dbgs(), BlockToFix, GE, loongarch::getEdgeKindName(*Kind));
      dbgs() << "\n";
    });

    BlockToFix.addEdge(std::move(GE));

    return Error::success();
  }

public:
  ELFLinkGraphBuilder_loongarch(StringRef FileName,
                                const object::ELFFile<ELFT> &Obj,
                                std::shared_ptr<orc::SymbolStringPool> SSP,
                                Triple TT, SubtargetFeatures Features)
      : ELFLinkGraphBuilder<ELFT>(Obj, std::move(SSP), std::move(TT),
                                  std::move(Features), FileName,
                                  loongarch::getEdgeKindName) {}
};

Error buildTables_ELF_loongarch(LinkGraph &G) {
  LLVM_DEBUG(dbgs() << "Visiting edges in graph:\n");

  GOTTableManager GOT;
  PLTTableManager PLT(GOT);
  visitExistingEdges(G, GOT, PLT);
  return Error::success();
}

} // namespace

namespace llvm {
namespace jitlink {

Expected<std::unique_ptr<LinkGraph>> createLinkGraphFromELFObject_loongarch(
    MemoryBufferRef ObjectBuffer, std::shared_ptr<orc::SymbolStringPool> SSP) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  auto ELFObj = object::ObjectFile::createELFObjectFile(ObjectBuffer);
  if (!ELFObj)
    return ELFObj.takeError();

  auto Features = (*ELFObj)->getFeatures();
  if (!Features)
    return Features.takeError();

  if ((*ELFObj)->getArch() == Triple::loongarch64) {
    auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF64LE>>(**ELFObj);
    return ELFLinkGraphBuilder_loongarch<object::ELF64LE>(
               (*ELFObj)->getFileName(), ELFObjFile.getELFFile(),
               std::move(SSP), (*ELFObj)->makeTriple(), std::move(*Features))
        .buildGraph();
  }

  assert((*ELFObj)->getArch() == Triple::loongarch32 &&
         "Invalid triple for LoongArch ELF object file");
  auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF32LE>>(**ELFObj);
  return ELFLinkGraphBuilder_loongarch<object::ELF32LE>(
             (*ELFObj)->getFileName(), ELFObjFile.getELFFile(), std::move(SSP),
             (*ELFObj)->makeTriple(), std::move(*Features))
      .buildGraph();
}

void link_ELF_loongarch(std::unique_ptr<LinkGraph> G,
                        std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  const Triple &TT = G->getTargetTriple();
  if (Ctx->shouldAddDefaultTargetPasses(TT)) {
    // Add eh-frame passes.
    Config.PrePrunePasses.push_back(DWARFRecordSectionSplitter(".eh_frame"));
    Config.PrePrunePasses.push_back(
        EHFrameEdgeFixer(".eh_frame", G->getPointerSize(), Pointer32, Pointer64,
                         Delta32, Delta64, NegDelta32));
    Config.PrePrunePasses.push_back(EHFrameNullTerminator(".eh_frame"));

    // Add a mark-live pass.
    if (auto MarkLive = Ctx->getMarkLivePass(TT))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);

    // Add an in-place GOT/PLTStubs build pass.
    Config.PostPrunePasses.push_back(buildTables_ELF_loongarch);

    // Add a linker relaxation pass.
    Config.PostAllocationPasses.push_back(relax);
  }

  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  ELFJITLinker_loongarch::link(std::move(Ctx), std::move(G), std::move(Config));
}

LinkGraphPassFunction createRelaxationPass_ELF_loongarch() { return relax; }

} // namespace jitlink
} // namespace llvm
