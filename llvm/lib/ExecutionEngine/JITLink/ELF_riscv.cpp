//===------- ELF_riscv.cpp -JIT linker implementation for ELF/riscv -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/riscv jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_riscv.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"
#include "PerGraphGOTAndPLTStubsBuilder.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/riscv.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"

#define DEBUG_TYPE "jitlink"
using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::riscv;

namespace {

class PerGraphGOTAndPLTStubsBuilder_ELF_riscv
    : public PerGraphGOTAndPLTStubsBuilder<
          PerGraphGOTAndPLTStubsBuilder_ELF_riscv> {
public:
  static constexpr size_t StubEntrySize = 16;
  static const uint8_t NullGOTEntryContent[8];
  static const uint8_t RV64StubContent[StubEntrySize];
  static const uint8_t RV32StubContent[StubEntrySize];

  using PerGraphGOTAndPLTStubsBuilder<
      PerGraphGOTAndPLTStubsBuilder_ELF_riscv>::PerGraphGOTAndPLTStubsBuilder;

  bool isRV64() const { return G.getPointerSize() == 8; }

  bool isGOTEdgeToFix(Edge &E) const { return E.getKind() == R_RISCV_GOT_HI20; }

  Symbol &createGOTEntry(Symbol &Target) {
    Block &GOTBlock =
        G.createContentBlock(getGOTSection(), getGOTEntryBlockContent(),
                             orc::ExecutorAddr(), G.getPointerSize(), 0);
    GOTBlock.addEdge(isRV64() ? R_RISCV_64 : R_RISCV_32, 0, Target, 0);
    return G.addAnonymousSymbol(GOTBlock, 0, G.getPointerSize(), false, false);
  }

  Symbol &createPLTStub(Symbol &Target) {
    Block &StubContentBlock = G.createContentBlock(
        getStubsSection(), getStubBlockContent(), orc::ExecutorAddr(), 4, 0);
    auto &GOTEntrySymbol = getGOTEntry(Target);
    StubContentBlock.addEdge(R_RISCV_CALL, 0, GOTEntrySymbol, 0);
    return G.addAnonymousSymbol(StubContentBlock, 0, StubEntrySize, true,
                                false);
  }

  void fixGOTEdge(Edge &E, Symbol &GOTEntry) {
    // Replace the relocation pair (R_RISCV_GOT_HI20, R_RISCV_PCREL_LO12)
    // with (R_RISCV_PCREL_HI20, R_RISCV_PCREL_LO12)
    // Therefore, here just change the R_RISCV_GOT_HI20 to R_RISCV_PCREL_HI20
    E.setKind(R_RISCV_PCREL_HI20);
    E.setTarget(GOTEntry);
  }

  void fixPLTEdge(Edge &E, Symbol &PLTStubs) {
    assert(E.getKind() == R_RISCV_CALL_PLT && "Not a R_RISCV_CALL_PLT edge?");
    E.setKind(R_RISCV_CALL);
    E.setTarget(PLTStubs);
  }

  bool isExternalBranchEdge(Edge &E) const {
    return E.getKind() == R_RISCV_CALL_PLT;
  }

private:
  Section &getGOTSection() const {
    if (!GOTSection)
      GOTSection = &G.createSection("$__GOT", MemProt::Read);
    return *GOTSection;
  }

  Section &getStubsSection() const {
    if (!StubsSection)
      StubsSection =
          &G.createSection("$__STUBS", MemProt::Read | MemProt::Exec);
    return *StubsSection;
  }

  ArrayRef<char> getGOTEntryBlockContent() {
    return {reinterpret_cast<const char *>(NullGOTEntryContent),
            G.getPointerSize()};
  }

  ArrayRef<char> getStubBlockContent() {
    auto StubContent = isRV64() ? RV64StubContent : RV32StubContent;
    return {reinterpret_cast<const char *>(StubContent), StubEntrySize};
  }

  mutable Section *GOTSection = nullptr;
  mutable Section *StubsSection = nullptr;
};

const uint8_t PerGraphGOTAndPLTStubsBuilder_ELF_riscv::NullGOTEntryContent[8] =
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

const uint8_t
    PerGraphGOTAndPLTStubsBuilder_ELF_riscv::RV64StubContent[StubEntrySize] = {
        0x17, 0x0e, 0x00, 0x00,  // auipc t3, literal
        0x03, 0x3e, 0x0e, 0x00,  // ld    t3, literal(t3)
        0x67, 0x00, 0x0e, 0x00,  // jr    t3
        0x13, 0x00, 0x00, 0x00}; // nop

const uint8_t
    PerGraphGOTAndPLTStubsBuilder_ELF_riscv::RV32StubContent[StubEntrySize] = {
        0x17, 0x0e, 0x00, 0x00,  // auipc t3, literal
        0x03, 0x2e, 0x0e, 0x00,  // lw    t3, literal(t3)
        0x67, 0x00, 0x0e, 0x00,  // jr    t3
        0x13, 0x00, 0x00, 0x00}; // nop
} // namespace
namespace llvm {
namespace jitlink {

static Expected<const Edge &> getRISCVPCRelHi20(const Edge &E) {
  using namespace riscv;
  assert((E.getKind() == R_RISCV_PCREL_LO12_I ||
          E.getKind() == R_RISCV_PCREL_LO12_S) &&
         "Can only have high relocation for R_RISCV_PCREL_LO12_I or "
         "R_RISCV_PCREL_LO12_S");

  const Symbol &Sym = E.getTarget();
  const Block &B = Sym.getBlock();
  orc::ExecutorAddrDiff Offset = Sym.getOffset();

  struct Comp {
    bool operator()(const Edge &Lhs, orc::ExecutorAddrDiff Offset) {
      return Lhs.getOffset() < Offset;
    }
    bool operator()(orc::ExecutorAddrDiff Offset, const Edge &Rhs) {
      return Offset < Rhs.getOffset();
    }
  };

  auto Bound =
      std::equal_range(B.edges().begin(), B.edges().end(), Offset, Comp{});

  for (auto It = Bound.first; It != Bound.second; ++It) {
    if (It->getKind() == R_RISCV_PCREL_HI20)
      return *It;
  }

  return make_error<JITLinkError>(
      "No HI20 PCREL relocation type be found for LO12 PCREL relocation type");
}

static uint32_t extractBits(uint32_t Num, unsigned Low, unsigned Size) {
  return (Num & (((1ULL << Size) - 1) << Low)) >> Low;
}

static inline bool isAlignmentCorrect(uint64_t Value, int N) {
  return (Value & (N - 1)) ? false : true;
}

// Requires 0 < N <= 64.
static inline bool isInRangeForImm(int64_t Value, int N) {
  return Value == llvm::SignExtend64(Value, N);
}

class ELFJITLinker_riscv : public JITLinker<ELFJITLinker_riscv> {
  friend class JITLinker<ELFJITLinker_riscv>;

public:
  ELFJITLinker_riscv(std::unique_ptr<JITLinkContext> Ctx,
                     std::unique_ptr<LinkGraph> G, PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    using namespace riscv;
    using namespace llvm::support;

    char *BlockWorkingMem = B.getAlreadyMutableContent().data();
    char *FixupPtr = BlockWorkingMem + E.getOffset();
    orc::ExecutorAddr FixupAddress = B.getAddress() + E.getOffset();
    switch (E.getKind()) {
    case R_RISCV_32: {
      int64_t Value = (E.getTarget().getAddress() + E.getAddend()).getValue();
      *(little32_t *)FixupPtr = static_cast<uint32_t>(Value);
      break;
    }
    case R_RISCV_64: {
      int64_t Value = (E.getTarget().getAddress() + E.getAddend()).getValue();
      *(little64_t *)FixupPtr = static_cast<uint64_t>(Value);
      break;
    }
    case R_RISCV_BRANCH: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      if (LLVM_UNLIKELY(!isInRangeForImm(Value >> 1, 12)))
        return makeTargetOutOfRangeError(G, B, E);
      if (LLVM_UNLIKELY(!isAlignmentCorrect(Value, 2)))
        return makeAlignmentError(FixupAddress, Value, 2, E);
      uint32_t Imm31_25 =
          extractBits(Value, 5, 6) << 25 | extractBits(Value, 12, 1) << 31;
      uint32_t Imm11_7 =
          extractBits(Value, 1, 4) << 8 | extractBits(Value, 11, 1) << 7;
      uint32_t RawInstr = *(little32_t *)FixupPtr;
      *(little32_t *)FixupPtr = (RawInstr & 0x1FFF07F) | Imm31_25 | Imm11_7;
      break;
    }
    case R_RISCV_JAL: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      if (LLVM_UNLIKELY(!isInRangeForImm(Value >> 1, 20)))
        return makeTargetOutOfRangeError(G, B, E);
      if (LLVM_UNLIKELY(!isAlignmentCorrect(Value, 2)))
        return makeAlignmentError(FixupAddress, Value, 2, E);
      uint32_t Imm20 = extractBits(Value, 20, 1) << 31;
      uint32_t Imm10_1 = extractBits(Value, 1, 10) << 21;
      uint32_t Imm11 = extractBits(Value, 11, 1) << 20;
      uint32_t Imm19_12 = extractBits(Value, 12, 8) << 12;
      uint32_t RawInstr = *(little32_t *)FixupPtr;
      *(little32_t *)FixupPtr = RawInstr | Imm20 | Imm10_1 | Imm11 | Imm19_12;
      break;
    }
    case R_RISCV_HI20: {
      int64_t Value = (E.getTarget().getAddress() + E.getAddend()).getValue();
      int64_t Hi = Value + 0x800;
      if (LLVM_UNLIKELY(!isInRangeForImm(Hi, 32)))
        return makeTargetOutOfRangeError(G, B, E);
      uint32_t RawInstr = *(little32_t *)FixupPtr;
      *(little32_t *)FixupPtr =
          (RawInstr & 0xFFF) | (static_cast<uint32_t>(Hi & 0xFFFFF000));
      break;
    }
    case R_RISCV_LO12_I: {
      // FIXME: We assume that R_RISCV_HI20 is present in object code and pairs
      // with current relocation R_RISCV_LO12_I. So here may need a check.
      int64_t Value = (E.getTarget().getAddress() + E.getAddend()).getValue();
      int32_t Lo = Value & 0xFFF;
      uint32_t RawInstr = *(little32_t *)FixupPtr;
      *(little32_t *)FixupPtr =
          (RawInstr & 0xFFFFF) | (static_cast<uint32_t>(Lo & 0xFFF) << 20);
      break;
    }
    case R_RISCV_CALL: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      int64_t Hi = Value + 0x800;
      if (LLVM_UNLIKELY(!isInRangeForImm(Hi, 32)))
        return makeTargetOutOfRangeError(G, B, E);
      int32_t Lo = Value & 0xFFF;
      uint32_t RawInstrAuipc = *(little32_t *)FixupPtr;
      uint32_t RawInstrJalr = *(little32_t *)(FixupPtr + 4);
      *(little32_t *)FixupPtr =
          RawInstrAuipc | (static_cast<uint32_t>(Hi & 0xFFFFF000));
      *(little32_t *)(FixupPtr + 4) =
          RawInstrJalr | (static_cast<uint32_t>(Lo) << 20);
      break;
    }
    case R_RISCV_PCREL_HI20: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      int64_t Hi = Value + 0x800;
      if (LLVM_UNLIKELY(!isInRangeForImm(Hi, 32)))
        return makeTargetOutOfRangeError(G, B, E);
      uint32_t RawInstr = *(little32_t *)FixupPtr;
      *(little32_t *)FixupPtr =
          (RawInstr & 0xFFF) | (static_cast<uint32_t>(Hi & 0xFFFFF000));
      break;
    }
    case R_RISCV_PCREL_LO12_I: {
      // FIXME: We assume that R_RISCV_PCREL_HI20 is present in object code and
      // pairs with current relocation R_RISCV_PCREL_LO12_I. So here may need a
      // check.
      auto RelHI20 = getRISCVPCRelHi20(E);
      if (!RelHI20)
        return RelHI20.takeError();
      int64_t Value = RelHI20->getTarget().getAddress() +
                      RelHI20->getAddend() - E.getTarget().getAddress();
      int64_t Lo = Value & 0xFFF;
      uint32_t RawInstr = *(little32_t *)FixupPtr;
      *(little32_t *)FixupPtr =
          (RawInstr & 0xFFFFF) | (static_cast<uint32_t>(Lo & 0xFFF) << 20);
      break;
    }
    case R_RISCV_PCREL_LO12_S: {
      // FIXME: We assume that R_RISCV_PCREL_HI20 is present in object code and
      // pairs with current relocation R_RISCV_PCREL_LO12_S. So here may need a
      // check.
      auto RelHI20 = getRISCVPCRelHi20(E);
      int64_t Value = RelHI20->getTarget().getAddress() +
                      RelHI20->getAddend() - E.getTarget().getAddress();
      int64_t Lo = Value & 0xFFF;
      uint32_t Imm31_25 = extractBits(Lo, 5, 7) << 25;
      uint32_t Imm11_7 = extractBits(Lo, 0, 5) << 7;
      uint32_t RawInstr = *(little32_t *)FixupPtr;

      *(little32_t *)FixupPtr = (RawInstr & 0x1FFF07F) | Imm31_25 | Imm11_7;
      break;
    }
    case R_RISCV_ADD64: {
      int64_t Value = (E.getTarget().getAddress() +
                       support::endian::read64le(reinterpret_cast<const void *>(
                           FixupAddress.getValue())) +
                       E.getAddend())
                          .getValue();
      *(little64_t *)FixupPtr = static_cast<uint64_t>(Value);
      break;
    }
    case R_RISCV_ADD32: {
      int64_t Value = (E.getTarget().getAddress() +
                       support::endian::read32le(reinterpret_cast<const void *>(
                           FixupAddress.getValue())) +
                       E.getAddend())
                          .getValue();
      *(little32_t *)FixupPtr = static_cast<uint32_t>(Value);
      break;
    }
    case R_RISCV_ADD16: {
      int64_t Value = (E.getTarget().getAddress() +
                       support::endian::read16le(reinterpret_cast<const void *>(
                           FixupAddress.getValue())) +
                       E.getAddend())
                          .getValue();
      *(little16_t *)FixupPtr = static_cast<uint32_t>(Value);
      break;
    }
    case R_RISCV_ADD8: {
      int64_t Value =
          (E.getTarget().getAddress() +
           *(reinterpret_cast<const uint8_t *>(FixupAddress.getValue())) +
           E.getAddend())
              .getValue();
      *FixupPtr = static_cast<uint8_t>(Value);
      break;
    }
    case R_RISCV_SUB64: {
      int64_t Value = support::endian::read64le(reinterpret_cast<const void *>(
                          FixupAddress.getValue())) -
                      E.getTarget().getAddress().getValue() - E.getAddend();
      *(little64_t *)FixupPtr = static_cast<uint64_t>(Value);
      break;
    }
    case R_RISCV_SUB32: {
      int64_t Value = support::endian::read32le(reinterpret_cast<const void *>(
                          FixupAddress.getValue())) -
                      E.getTarget().getAddress().getValue() - E.getAddend();
      *(little32_t *)FixupPtr = static_cast<uint32_t>(Value);
      break;
    }
    case R_RISCV_SUB16: {
      int64_t Value = support::endian::read16le(reinterpret_cast<const void *>(
                          FixupAddress.getValue())) -
                      E.getTarget().getAddress().getValue() - E.getAddend();
      *(little16_t *)FixupPtr = static_cast<uint32_t>(Value);
      break;
    }
    case R_RISCV_SUB8: {
      int64_t Value =
          *(reinterpret_cast<const uint8_t *>(FixupAddress.getValue())) -
          E.getTarget().getAddress().getValue() - E.getAddend();
      *FixupPtr = static_cast<uint8_t>(Value);
      break;
    }
    case R_RISCV_SUB6: {
      int64_t Value =
          *(reinterpret_cast<const uint8_t *>(FixupAddress.getValue())) & 0x3f;
      Value -= E.getTarget().getAddress().getValue() - E.getAddend();
      *FixupPtr = (*FixupPtr & 0xc0) | (static_cast<uint8_t>(Value) & 0x3f);
      break;
    }
    case R_RISCV_SET6: {
      int64_t Value = (E.getTarget().getAddress() + E.getAddend()).getValue();
      uint32_t RawData = *(little32_t *)FixupPtr;
      int64_t Word6 = Value & 0x3f;
      *(little32_t *)FixupPtr = (RawData & 0xffffffc0) | Word6;
      break;
    }
    case R_RISCV_SET8: {
      int64_t Value = (E.getTarget().getAddress() + E.getAddend()).getValue();
      uint32_t RawData = *(little32_t *)FixupPtr;
      int64_t Word8 = Value & 0xff;
      *(little32_t *)FixupPtr = (RawData & 0xffffff00) | Word8;
      break;
    }
    case R_RISCV_SET16: {
      int64_t Value = (E.getTarget().getAddress() + E.getAddend()).getValue();
      uint32_t RawData = *(little32_t *)FixupPtr;
      int64_t Word16 = Value & 0xffff;
      *(little32_t *)FixupPtr = (RawData & 0xffff0000) | Word16;
      break;
    }
    case R_RISCV_SET32: {
      int64_t Value = (E.getTarget().getAddress() + E.getAddend()).getValue();
      int64_t Word32 = Value & 0xffffffff;
      *(little32_t *)FixupPtr = Word32;
      break;
    }
    case R_RISCV_32_PCREL: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      int64_t Word32 = Value & 0xffffffff;
      *(little32_t *)FixupPtr = Word32;
      break;
    }
    }
    return Error::success();
  }
};

template <typename ELFT>
class ELFLinkGraphBuilder_riscv : public ELFLinkGraphBuilder<ELFT> {
private:
  static Expected<riscv::EdgeKind_riscv>
  getRelocationKind(const uint32_t Type) {
    using namespace riscv;
    switch (Type) {
    case ELF::R_RISCV_32:
      return EdgeKind_riscv::R_RISCV_32;
    case ELF::R_RISCV_64:
      return EdgeKind_riscv::R_RISCV_64;
    case ELF::R_RISCV_BRANCH:
      return EdgeKind_riscv::R_RISCV_BRANCH;
    case ELF::R_RISCV_JAL:
      return EdgeKind_riscv::R_RISCV_JAL;
    case ELF::R_RISCV_HI20:
      return EdgeKind_riscv::R_RISCV_HI20;
    case ELF::R_RISCV_LO12_I:
      return EdgeKind_riscv::R_RISCV_LO12_I;
    case ELF::R_RISCV_CALL:
      return EdgeKind_riscv::R_RISCV_CALL;
    case ELF::R_RISCV_PCREL_HI20:
      return EdgeKind_riscv::R_RISCV_PCREL_HI20;
    case ELF::R_RISCV_PCREL_LO12_I:
      return EdgeKind_riscv::R_RISCV_PCREL_LO12_I;
    case ELF::R_RISCV_PCREL_LO12_S:
      return EdgeKind_riscv::R_RISCV_PCREL_LO12_S;
    case ELF::R_RISCV_GOT_HI20:
      return EdgeKind_riscv::R_RISCV_GOT_HI20;
    case ELF::R_RISCV_CALL_PLT:
      return EdgeKind_riscv::R_RISCV_CALL_PLT;
    case ELF::R_RISCV_ADD64:
      return EdgeKind_riscv::R_RISCV_ADD64;
    case ELF::R_RISCV_ADD32:
      return EdgeKind_riscv::R_RISCV_ADD32;
    case ELF::R_RISCV_ADD16:
      return EdgeKind_riscv::R_RISCV_ADD16;
    case ELF::R_RISCV_ADD8:
      return EdgeKind_riscv::R_RISCV_ADD8;
    case ELF::R_RISCV_SUB64:
      return EdgeKind_riscv::R_RISCV_SUB64;
    case ELF::R_RISCV_SUB32:
      return EdgeKind_riscv::R_RISCV_SUB32;
    case ELF::R_RISCV_SUB16:
      return EdgeKind_riscv::R_RISCV_SUB16;
    case ELF::R_RISCV_SUB8:
      return EdgeKind_riscv::R_RISCV_SUB8;
    case ELF::R_RISCV_SUB6:
      return EdgeKind_riscv::R_RISCV_SUB6;
    case ELF::R_RISCV_SET6:
      return EdgeKind_riscv::R_RISCV_SET6;
    case ELF::R_RISCV_SET8:
      return EdgeKind_riscv::R_RISCV_SET8;
    case ELF::R_RISCV_SET16:
      return EdgeKind_riscv::R_RISCV_SET16;
    case ELF::R_RISCV_SET32:
      return EdgeKind_riscv::R_RISCV_SET32;
    case ELF::R_RISCV_32_PCREL:
      return EdgeKind_riscv::R_RISCV_32_PCREL;
    }

    return make_error<JITLinkError>(
        "Unsupported riscv relocation:" + formatv("{0:d}: ", Type) +
        object::getELFRelocationTypeName(ELF::EM_RISCV, Type));
  }

  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Processing relocations:\n");

    using Base = ELFLinkGraphBuilder<ELFT>;
    using Self = ELFLinkGraphBuilder_riscv<ELFT>;
    for (const auto &RelSect : Base::Sections)
      if (Error Err = Base::forEachRelocation(RelSect, this,
                                              &Self::addSingleRelocation))
        return Err;

    return Error::success();
  }

  Error addSingleRelocation(const typename ELFT::Rela &Rel,
                            const typename ELFT::Shdr &FixupSect,
                            Block &BlockToFix) {
    using Base = ELFLinkGraphBuilder<ELFT>;

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

    uint32_t Type = Rel.getType(false);
    Expected<riscv::EdgeKind_riscv> Kind = getRelocationKind(Type);
    if (!Kind)
      return Kind.takeError();

    int64_t Addend = Rel.r_addend;
    auto FixupAddress = orc::ExecutorAddr(FixupSect.sh_addr) + Rel.r_offset;
    Edge::OffsetT Offset = FixupAddress - BlockToFix.getAddress();
    Edge GE(*Kind, Offset, *GraphSymbol, Addend);
    LLVM_DEBUG({
      dbgs() << "    ";
      printEdge(dbgs(), BlockToFix, GE, riscv::getEdgeKindName(*Kind));
      dbgs() << "\n";
    });

    BlockToFix.addEdge(std::move(GE));
    return Error::success();
  }

public:
  ELFLinkGraphBuilder_riscv(StringRef FileName,
                            const object::ELFFile<ELFT> &Obj, const Triple T)
      : ELFLinkGraphBuilder<ELFT>(Obj, std::move(T), FileName,
                                  riscv::getEdgeKindName) {}
};

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_riscv(MemoryBufferRef ObjectBuffer) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  auto ELFObj = object::ObjectFile::createELFObjectFile(ObjectBuffer);
  if (!ELFObj)
    return ELFObj.takeError();

  if ((*ELFObj)->getArch() == Triple::riscv64) {
    auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF64LE>>(**ELFObj);
    return ELFLinkGraphBuilder_riscv<object::ELF64LE>(
               (*ELFObj)->getFileName(), ELFObjFile.getELFFile(),
               (*ELFObj)->makeTriple())
        .buildGraph();
  } else {
    assert((*ELFObj)->getArch() == Triple::riscv32 &&
           "Invalid triple for RISCV ELF object file");
    auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF32LE>>(**ELFObj);
    return ELFLinkGraphBuilder_riscv<object::ELF32LE>(
               (*ELFObj)->getFileName(), ELFObjFile.getELFFile(),
               (*ELFObj)->makeTriple())
        .buildGraph();
  }
}

void link_ELF_riscv(std::unique_ptr<LinkGraph> G,
                    std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  const Triple &TT = G->getTargetTriple();
  if (Ctx->shouldAddDefaultTargetPasses(TT)) {
    if (auto MarkLive = Ctx->getMarkLivePass(TT))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);
    Config.PostPrunePasses.push_back(
        PerGraphGOTAndPLTStubsBuilder_ELF_riscv::asPass);
  }
  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  ELFJITLinker_riscv::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace jitlink
} // namespace llvm
