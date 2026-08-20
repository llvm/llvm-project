//===-- ELF_mips.cpp - JIT linker implementation for ELF/MIPS ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_mips.h"

#include "EHFrameSupportImpl.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/DWARFRecordSectionSplitter.h"
#include "llvm/ExecutionEngine/JITLink/mips.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MipsABIFlags.h"
#include "llvm/Support/raw_ostream.h"

#include <tuple>

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::mips;

namespace {

constexpr StringRef GOTSectionName = "$__GOT";
constexpr StringRef StubsSectionName = "$__STUBS";
constexpr StringRef TLSInfoSectionName = "$__TLSINFO";
constexpr StringRef TLSBaseName = "__jitlink_mips_tls_base";
constexpr StringRef GOTSymbolName = "_GLOBAL_OFFSET_TABLE_";
constexpr StringRef GPName = "_gp";
constexpr StringRef GPDispName = "_gp_disp";
constexpr StringRef LocalGPName = "__gnu_local_gp";
constexpr unsigned InstructionSize = sizeof(uint32_t);
constexpr unsigned RelocationTypeBits = 8;
constexpr unsigned MaxRelocationOperations = 3;
constexpr unsigned GOTPageBits = 16;
constexpr uint64_t GOTPageSize = UINT64_C(1) << GOTPageBits;
constexpr uint64_t GOTPageMask = GOTPageSize - 1;
constexpr uint64_t GOTPageBias = GOTPageSize / 2;
constexpr unsigned JumpRegionBits = 28;
constexpr uint64_t JumpRegionMask = ~maskTrailingOnes<uint64_t>(JumpRegionBits);
constexpr int64_t CompactBranchPCBias = InstructionSize;
// The MIPS psABI defines $gp as 0x7ff0 bytes past the GOT base.
constexpr uint64_t GPAnchorOffset = 0x7ff0;

enum class MipsABI { O32, N32, N64 };

static uint16_t getLo16(uint64_t Value) { return static_cast<uint16_t>(Value); }

static uint16_t getHi16(uint64_t Value) {
  return static_cast<uint16_t>((Value + GOTPageBias) >> 16);
}

static uint16_t getHigher16(uint64_t Value) {
  constexpr uint64_t Bias = GOTPageBias | (GOTPageBias << 16);
  return static_cast<uint16_t>((Value + Bias) >> 32);
}

static uint16_t getHighest16(uint64_t Value) {
  constexpr uint64_t Bias =
      GOTPageBias | (GOTPageBias << 16) | (GOTPageBias << 32);
  return static_cast<uint16_t>((Value + Bias) >> 48);
}

static uint64_t getGOTPage(uint64_t Value) {
  return (Value + GOTPageBias) & ~GOTPageMask;
}

static Edge::Kind getPagePointerEdgeKind(const LinkGraph &G) {
  return G.getPointerSize() == 8 ? PagePointer64 : PagePointer32;
}

struct PCRelEncoding {
  unsigned Bits;
  unsigned Shift;
  unsigned PCAlignment;
};

static PCRelEncoding getPCRelEncoding(Edge::Kind Kind) {
  switch (Kind) {
  case PC16:
    return {16, 2, 1};
  case PC18S3:
    return {18, 3, 8};
  case PC19S2:
    return {19, 2, 4};
  case PC21S2:
    return {21, 2, 1};
  case PC26S2:
    return {26, 2, 1};
  default:
    llvm_unreachable("not a MIPS immediate branch edge");
  }
}

template <unsigned Bits, unsigned Shift>
static int64_t decodeInstructionImmediate(uint32_t Instruction) {
  static_assert(Bits + Shift <= 32);
  uint32_t Value = (Instruction & maskTrailingOnes<uint32_t>(Bits)) << Shift;
  return SignExtend64<Bits + Shift>(Value);
}

static uint8_t getPackedRelocationField(uint32_t Packed, unsigned Index) {
  return static_cast<uint8_t>((Packed >> (Index * RelocationTypeBits)) &
                              maskTrailingOnes<uint32_t>(RelocationTypeBits));
}

static std::string getRelocationChainName(ArrayRef<uint8_t> Types) {
  std::string Result;
  raw_string_ostream OS(Result);
  for (auto [Index, Type] : llvm::enumerate(Types)) {
    if (Index)
      OS << '/';
    OS << object::getELFRelocationTypeName(ELF::EM_MIPS, Type);
  }
  return Result;
}

static bool needsGP(Edge::Kind K) {
  switch (K) {
  case GPRel16:
  case GPRel32:
  case GPRel64:
  case GOTOffset16:
  case GOTOffsetHi16:
  case GOTOffsetLo16:
  case GPDispHi16:
  case GPDispLo16:
  case NegGPRelHi16:
  case NegGPRelLo16:
    return true;
  default:
    return false;
  }
}

static unsigned getFixupSize(Edge::Kind K) {
  switch (K) {
  case Abs16:
    return sizeof(uint16_t);
  case Pointer64:
  case PagePointer64:
  case Delta64:
  case GPRel64:
  case DTPRel64:
    return sizeof(uint64_t);
  case Pointer32:
  case PagePointer32:
  case Delta32:
  case NegDelta32:
  case Hi16:
  case Lo16:
  case Higher16:
  case Highest16:
  case Jump26:
  case PC16:
  case PC32:
  case PC18S3:
  case PC19S2:
  case PC21S2:
  case PC26S2:
  case PCHi16:
  case PCLo16:
  case GPDispHi16:
  case GPDispLo16:
  case GPRel16:
  case GPRel32:
  case GOTOffset16:
  case GOTOffsetHi16:
  case GOTOffsetLo16:
  case GOTPageOffset16:
  case DTPRelHi16:
  case DTPRelLo16:
  case DTPRel32:
  case NegGPRelHi16:
  case NegGPRelLo16:
  case RequestGOTAndTransformToOffset16:
  case RequestGOTPageAndTransformToOffset16:
  case RequestGOTAndTransformToOffsetHi16:
  case RequestGOTAndTransformToOffsetLo16:
  case RequestTLSGDAndTransformToOffset16:
  case RequestTLSLDMAndTransformToOffset16:
    return sizeof(uint32_t);
  default:
    llvm_unreachable("not a MIPS relocation edge");
  }
}

class ELFJITLinker_mips : public JITLinker<ELFJITLinker_mips> {
  friend class JITLinker<ELFJITLinker_mips>;

public:
  ELFJITLinker_mips(std::unique_ptr<JITLinkContext> Ctx,
                    std::unique_ptr<LinkGraph> G, PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    char *Fixup = B.getAlreadyMutableContent().data() + E.getOffset();
    uint64_t P = B.getFixupAddress(E).getValue();
    uint64_t S = E.getTarget().getAddress().getValue();
    int64_t A = E.getAddend();
    uint64_t TargetAddress = S + A;
    const endianness Endianness = G.getEndianness();

    auto TLSBaseAddr = [&]() -> Expected<uint64_t> {
      auto Name = G.intern(TLSBaseName);
      if (auto *Base = G.findDefinedSymbolByName(Name))
        return Base->getAddress().getValue();
      return make_error<JITLinkError>(
          "MIPS DTPREL relocation requires a TLS template");
    };
    auto CheckSigned = [&](int64_t V, unsigned Bits) -> Error {
      if (!isIntN(Bits, V))
        return makeTargetOutOfRangeError(G, B, E);
      return Error::success();
    };
    auto CheckAligned = [&](int64_t V, unsigned Align) -> Error {
      if (V & (Align - 1))
        return makeAlignmentError(orc::ExecutorAddr(P), V, Align, E);
      return Error::success();
    };

    int64_t V = static_cast<int64_t>(TargetAddress);
    std::optional<uint64_t> GP;
    if (needsGP(E.getKind())) {
      auto *GPSymbol = G.findDefinedSymbolByName(G.intern(GPName));
      assert(GPSymbol && "missing MIPS _gp symbol");
      GP = GPSymbol->getAddress().getValue();
    }

    switch (E.getKind()) {
    case Pointer32:
      if (TargetAddress > UINT32_MAX)
        return makeTargetOutOfRangeError(G, B, E);
      support::endian::write32(Fixup, static_cast<uint32_t>(TargetAddress),
                               Endianness);
      break;
    case Pointer64:
      support::endian::write64(Fixup, TargetAddress, Endianness);
      break;
    case PagePointer32: {
      uint64_t Page = getGOTPage(TargetAddress);
      if (Page > UINT32_MAX)
        return makeTargetOutOfRangeError(G, B, E);
      support::endian::write32(Fixup, Page, Endianness);
      break;
    }
    case PagePointer64:
      support::endian::write64(Fixup, getGOTPage(TargetAddress), Endianness);
      break;
    case Delta32:
    case PC32:
      V = static_cast<int64_t>(TargetAddress - P);
      if (auto Err = CheckSigned(V, 32))
        return Err;
      support::endian::write32(Fixup, V, Endianness);
      break;
    case Delta64:
      support::endian::write64(Fixup, TargetAddress - P, Endianness);
      break;
    case NegDelta32:
      V = static_cast<int64_t>(P - S + A);
      if (auto Err = CheckSigned(V, 32))
        return Err;
      support::endian::write32(Fixup, V, Endianness);
      break;
    case Abs16:
      if (auto Err = CheckSigned(V, 16))
        return Err;
      support::endian::write16(Fixup, V, Endianness);
      break;
    case Hi16:
      writeImmediate16(Fixup, getHi16(TargetAddress), Endianness);
      break;
    case Lo16:
      writeImmediate16(Fixup, getLo16(TargetAddress), Endianness);
      break;
    case Higher16:
      writeImmediate16(Fixup, getHigher16(TargetAddress), Endianness);
      break;
    case Highest16:
      writeImmediate16(Fixup, getHighest16(TargetAddress), Endianness);
      break;
    case Jump26: {
      if (auto Err = CheckAligned(TargetAddress, InstructionSize))
        return Err;
      if (((P + InstructionSize) & JumpRegionMask) !=
          (TargetAddress & JumpRegionMask))
        return makeTargetOutOfRangeError(G, B, E);
      writeImmediate26(Fixup, TargetAddress >> 2, Endianness);
      break;
    }
    case PC16:
    case PC18S3:
    case PC19S2:
    case PC21S2:
    case PC26S2: {
      PCRelEncoding Encoding = getPCRelEncoding(E.getKind());
      uint64_t FixupPC = alignDown(P, Encoding.PCAlignment);
      V = static_cast<int64_t>(TargetAddress - FixupPC);
      if (auto Err = CheckAligned(V, 1U << Encoding.Shift))
        return Err;
      if (auto Err = CheckSigned(V, Encoding.Bits + Encoding.Shift))
        return Err;
      uint32_t Mask = maskTrailingOnes<uint32_t>(Encoding.Bits);
      writeMaskedInstruction32(
          Fixup, Mask, static_cast<uint64_t>(V) >> Encoding.Shift, Endianness);
      break;
    }
    case PCHi16:
      V = static_cast<int64_t>(TargetAddress - P);
      writeImmediate16(Fixup, getHi16(V), Endianness);
      break;
    case PCLo16:
      writeImmediate16(Fixup, getLo16(TargetAddress - P), Endianness);
      break;
    case GPDispHi16:
      V = static_cast<int64_t>(*GP + A - P);
      writeImmediate16(Fixup, getHi16(V), Endianness);
      break;
    case GPDispLo16:
      // Both halves use the address of the high instruction as P.
      writeImmediate16(Fixup, getLo16(*GP + A - P + InstructionSize),
                       Endianness);
      break;
    case GPRel16:
    case GOTOffset16:
      V = static_cast<int64_t>(S + A - *GP);
      if (auto Err = CheckSigned(V, 16))
        return Err;
      writeImmediate16(Fixup, V, Endianness);
      break;
    case GPRel32:
      V = static_cast<int64_t>(S + A - *GP);
      if (auto Err = CheckSigned(V, 32))
        return Err;
      support::endian::write32(Fixup, V, Endianness);
      break;
    case GPRel64:
      support::endian::write64(Fixup, TargetAddress - *GP, Endianness);
      break;
    case GOTOffsetHi16:
      V = static_cast<int64_t>(S + A - *GP);
      writeImmediate16(Fixup, getHi16(V), Endianness);
      break;
    case GOTOffsetLo16:
      writeImmediate16(Fixup, getLo16(TargetAddress - *GP), Endianness);
      break;
    case GOTPageOffset16: {
      uint64_t Page = getGOTPage(TargetAddress);
      writeImmediate16(Fixup, getLo16(TargetAddress - Page), Endianness);
      break;
    }
    case DTPRelHi16:
    case DTPRelLo16:
    case DTPRel32:
    case DTPRel64: {
      auto BaseOrErr = TLSBaseAddr();
      if (!BaseOrErr)
        return BaseOrErr.takeError();
      V = static_cast<int64_t>(S + A - *BaseOrErr);
      if (E.getKind() == DTPRelHi16)
        writeImmediate16(Fixup, getHi16(V), Endianness);
      else if (E.getKind() == DTPRelLo16)
        writeImmediate16(Fixup, getLo16(V), Endianness);
      else if (E.getKind() == DTPRel32) {
        if (auto Err = CheckSigned(V, 32))
          return Err;
        support::endian::write32(Fixup, V, Endianness);
      } else
        support::endian::write64(Fixup, V, Endianness);
      break;
    }
    case NegGPRelHi16:
    case NegGPRelLo16:
      V = static_cast<int64_t>(*GP - S - A);
      if (E.getKind() == NegGPRelHi16)
        writeImmediate16(Fixup, getHi16(V), Endianness);
      else
        writeImmediate16(Fixup, getLo16(V), Endianness);
      break;
    default:
      return make_error<JITLinkError>(
          "In graph " + G.getName() + ", section " + B.getSection().getName() +
          ": unsupported MIPS edge kind " + G.getEdgeKindName(E.getKind()));
    }
    return Error::success();
  }
};

template <typename ELFT>
class ELFLinkGraphBuilder_mips : public ELFLinkGraphBuilder<ELFT> {
  using Base = ELFLinkGraphBuilder<ELFT>;
  using Rel = typename ELFT::Rel;
  using Rela = typename ELFT::Rela;

  struct Reloc {
    uint64_t Offset;
    uint32_t Symbol;
    SmallVector<uint8_t, MaxRelocationOperations> Types;
    int64_t Addend;
  };

public:
  ELFLinkGraphBuilder_mips(StringRef FileName, const object::ELFFile<ELFT> &Obj,
                           std::shared_ptr<orc::SymbolStringPool> SSP,
                           Triple TT, SubtargetFeatures Features, MipsABI ABI)
      : Base(Obj, std::move(SSP), std::move(TT), std::move(Features), FileName,
             mips::getEdgeKindName),
        ABI(ABI) {}

private:
  MipsABI ABI;

  bool excludeSection(const typename ELFT::Shdr &Sec) const override {
    return Sec.sh_type == ELF::SHT_MIPS_ABIFLAGS ||
           Sec.sh_type == ELF::SHT_MIPS_REGINFO ||
           Sec.sh_type == ELF::SHT_MIPS_OPTIONS;
  }

  Error error(const typename ELFT::Shdr &FixupSect, uint64_t Offset,
              const Twine &Message) const {
    auto Name = Base::Obj.getSectionName(FixupSect);
    if (!Name)
      return Name.takeError();
    return make_error<JITLinkError>(Base::G->getName() + ": section " + *Name +
                                    "+0x" + utohexstr(Offset) + ": " + Message);
  }

  Expected<int64_t> implicitAddend(const typename ELFT::Shdr &RelSect,
                                   uint8_t Type, const Block &B,
                                   uint64_t Offset) const {
    if (Type == ELF::R_MIPS_NONE || Type == ELF::R_MIPS_JALR)
      return 0;

    unsigned FixupSize = Type == ELF::R_MIPS_16   ? sizeof(uint16_t)
                         : Type == ELF::R_MIPS_64 ? sizeof(uint64_t)
                                                  : sizeof(uint32_t);
    if (Offset > B.getSize() || FixupSize > B.getSize() - Offset)
      return error(RelSect, Offset,
                   "relocation fixup extends past the end of its block");

    const char *P = B.getContent().data() + Offset;
    if (Type == ELF::R_MIPS_16)
      return SignExtend64<16>(support::endian::read16<ELFT::Endianness>(P));

    if (Type == ELF::R_MIPS_64)
      return support::endian::read64<ELFT::Endianness>(P);

    uint32_t W = support::endian::read32<ELFT::Endianness>(P);
    switch (Type) {
    case ELF::R_MIPS_32:
    case ELF::R_MIPS_GPREL32:
    case ELF::R_MIPS_PC32:
    case ELF::R_MIPS_TLS_DTPREL32:
      return SignExtend64<32>(W);
    case ELF::R_MIPS_26:
      return decodeInstructionImmediate<26, 2>(W);
    case ELF::R_MIPS_HI16:
    case ELF::R_MIPS_PCHI16:
    case ELF::R_MIPS_GOT16:
    case ELF::R_MIPS_GOT_HI16:
    case ELF::R_MIPS_CALL_HI16:
      return decodeInstructionImmediate<16, 16>(W);
    case ELF::R_MIPS_PC16:
      return decodeInstructionImmediate<16, 2>(W);
    case ELF::R_MIPS_PC18_S3:
      return decodeInstructionImmediate<18, 3>(W);
    case ELF::R_MIPS_PC19_S2:
      return decodeInstructionImmediate<19, 2>(W);
    case ELF::R_MIPS_PC21_S2:
      return decodeInstructionImmediate<21, 2>(W);
    case ELF::R_MIPS_PC26_S2:
      return decodeInstructionImmediate<26, 2>(W);
    default:
      return decodeInstructionImmediate<16, 0>(W);
    }
  }

  static uint8_t matchingLow(uint8_t Type, bool IsLocal) {
    switch (Type) {
    case ELF::R_MIPS_HI16:
      return ELF::R_MIPS_LO16;
    case ELF::R_MIPS_PCHI16:
      return ELF::R_MIPS_PCLO16;
    case ELF::R_MIPS_GOT16:
      return IsLocal ? ELF::R_MIPS_LO16 : ELF::R_MIPS_NONE;
    default:
      return ELF::R_MIPS_NONE;
    }
  }

  Error validateSymbols() {
    if (!Base::SymTabSec)
      return Error::success();
    auto Symbols = Base::Obj.symbols(Base::SymTabSec);
    if (!Symbols)
      return Symbols.takeError();
    for (const auto &Sym : *Symbols)
      if ((Sym.st_other & ELF::STO_MIPS_MICROMIPS) ||
          (Sym.st_other & ELF::STO_MIPS_MIPS16) == ELF::STO_MIPS_MIPS16)
        return make_error<JITLinkError>(
            Base::G->getName() +
            ": compact-mode MIPS16/microMIPS symbols are unsupported");
    return Error::success();
  }

  Error collectRelocs(const typename ELFT::Shdr &RelSect, Block &B,
                      SmallVectorImpl<Reloc> &Out) {
    if (RelSect.sh_type == ELF::SHT_RELA) {
      auto Rs = Base::Obj.relas(RelSect);
      if (!Rs)
        return Rs.takeError();
      for (auto I = Rs->begin(), E = Rs->end(); I != E;) {
        const Rela &R = *I++;
        if (ABI == MipsABI::N32) {
          Reloc RR{R.r_offset, R.getSymbol(false), {}, R.r_addend};
          RR.Types.push_back(R.getType(false));
          while (I != E && I->r_offset == R.r_offset) {
            if (RR.Types.size() == MaxRelocationOperations)
              return error(RelSect, R.r_offset,
                           "more than three N32 relocation operations");
            if (I->getSymbol(false) != 0)
              return error(RelSect, R.r_offset,
                           "non-zero symbol in a secondary N32 relocation");
            RR.Types.push_back(I->getType(false));
            ++I;
          }
          Out.push_back(std::move(RR));
        } else {
          uint32_t PackedType = R.getType(Base::Obj.isMips64EL());
          Reloc RR{
              R.r_offset, R.getSymbol(Base::Obj.isMips64EL()), {}, R.r_addend};
          uint8_t T2 = getPackedRelocationField(PackedType, 1);
          uint8_t T3 = getPackedRelocationField(PackedType, 2);
          RR.Types.push_back(getPackedRelocationField(PackedType, 0));
          if (T2 || T3)
            RR.Types.push_back(T2);
          if (T3)
            RR.Types.push_back(T3);
          uint8_t SSym = getPackedRelocationField(PackedType, 3);
          if (SSym != ELF::RSS_UNDEF)
            return error(RelSect, R.r_offset,
                         "unsupported packed relocation special symbol " +
                             Twine(SSym));
          Out.push_back(std::move(RR));
        }
      }
      return Error::success();
    }

    auto Rs = Base::Obj.rels(RelSect);
    if (!Rs)
      return Rs.takeError();
    for (auto I = Rs->begin(), E = Rs->end(); I != E;) {
      const Rel &R = *I++;
      Reloc RR{R.r_offset, R.getSymbol(false), {}, 0};
      RR.Types.push_back(R.getType(false));
      if (ABI == MipsABI::N32) {
        while (I != E && I->r_offset == R.r_offset) {
          if (RR.Types.size() == MaxRelocationOperations)
            return error(RelSect, R.r_offset,
                         "more than three N32 relocation operations");
          if (I->getSymbol(false) != 0)
            return error(RelSect, R.r_offset,
                         "non-zero symbol in a secondary N32 relocation");
          RR.Types.push_back(I->getType(false));
          ++I;
        }
      }
      Out.push_back(std::move(RR));
    }

    for (size_t I = 0; I != Out.size(); ++I) {
      auto &R = Out[I];
      uint8_t FinalType = R.Types.back();
      auto Addend = implicitAddend(RelSect, FinalType, B, R.Offset);
      if (!Addend)
        return Addend.takeError();
      R.Addend = *Addend;
      bool IsLocal = false;
      if (auto *S = Base::getGraphSymbol(R.Symbol))
        IsLocal = S->getScope() == Scope::Local;
      uint8_t LowType = matchingLow(R.Types.front(), IsLocal);
      if (LowType == ELF::R_MIPS_NONE)
        continue;
      bool Found = false;
      for (size_t J = I + 1; J != Out.size(); ++J)
        if (Out[J].Types.front() == LowType && Out[J].Symbol == R.Symbol) {
          auto LowAddend = implicitAddend(RelSect, LowType, B, Out[J].Offset);
          if (!LowAddend)
            return LowAddend.takeError();
          R.Addend += *LowAddend;
          Found = true;
          break;
        }
      if (!Found)
        return error(RelSect, R.Offset,
                     "unmatched " +
                         Twine(object::getELFRelocationTypeName(
                             ELF::EM_MIPS, R.Types.front())) +
                         " relocation");
    }
    return Error::success();
  }

  Expected<Edge::Kind> edgeKind(const Reloc &R, Symbol &Target) const {
    if (R.Types.size() > 1) {
      if (R.Types.size() == 2 && R.Types[1] == ELF::R_MIPS_64) {
        switch (R.Types[0]) {
        case ELF::R_MIPS_32:
        case ELF::R_MIPS_64:
          return Pointer64;
        case ELF::R_MIPS_PC32:
          return Delta64;
        case ELF::R_MIPS_GPREL32:
          return GPRel64;
        default:
          break;
        }
      }
      if (R.Types.size() == 3 && R.Types[0] == ELF::R_MIPS_GPREL16 &&
          R.Types[1] == ELF::R_MIPS_SUB && R.Types[2] == ELF::R_MIPS_HI16)
        return NegGPRelHi16;
      if (R.Types.size() == 3 && R.Types[0] == ELF::R_MIPS_GPREL16 &&
          R.Types[1] == ELF::R_MIPS_SUB && R.Types[2] == ELF::R_MIPS_LO16)
        return NegGPRelLo16;
      return make_error<JITLinkError>("unsupported packed MIPS relocation " +
                                      getRelocationChainName(R.Types));
    }

    StringRef Name = Target.hasName() ? *Target.getName() : StringRef();
    switch (R.Types.front()) {
    case ELF::R_MIPS_16:
      return Abs16;
    case ELF::R_MIPS_32:
      return Pointer32;
    case ELF::R_MIPS_64:
      return Pointer64;
    case ELF::R_MIPS_HI16:
      return Name == GPDispName ? GPDispHi16 : Hi16;
    case ELF::R_MIPS_LO16:
      return Name == GPDispName ? GPDispLo16 : Lo16;
    case ELF::R_MIPS_HIGHER:
      return Higher16;
    case ELF::R_MIPS_HIGHEST:
      return Highest16;
    case ELF::R_MIPS_26:
      return Jump26;
    case ELF::R_MIPS_PC16:
      return PC16;
    case ELF::R_MIPS_PC32:
      return PC32;
    case ELF::R_MIPS_PC18_S3:
      return PC18S3;
    case ELF::R_MIPS_PC19_S2:
      return PC19S2;
    case ELF::R_MIPS_PC21_S2:
      return PC21S2;
    case ELF::R_MIPS_PC26_S2:
      return PC26S2;
    case ELF::R_MIPS_PCHI16:
      return PCHi16;
    case ELF::R_MIPS_PCLO16:
      return PCLo16;
    case ELF::R_MIPS_GPREL16:
      return GPRel16;
    case ELF::R_MIPS_GPREL32:
      return GPRel32;
    case ELF::R_MIPS_GOT16:
      return Target.getScope() == Scope::Local
                 ? RequestGOTPageAndTransformToOffset16
                 : RequestGOTAndTransformToOffset16;
    case ELF::R_MIPS_CALL16:
    case ELF::R_MIPS_GOT_DISP:
      return RequestGOTAndTransformToOffset16;
    case ELF::R_MIPS_GOT_PAGE:
      return RequestGOTPageAndTransformToOffset16;
    case ELF::R_MIPS_GOT_OFST:
      return GOTPageOffset16;
    case ELF::R_MIPS_GOT_HI16:
    case ELF::R_MIPS_CALL_HI16:
      return RequestGOTAndTransformToOffsetHi16;
    case ELF::R_MIPS_GOT_LO16:
    case ELF::R_MIPS_CALL_LO16:
      return RequestGOTAndTransformToOffsetLo16;
    case ELF::R_MIPS_TLS_GD:
      return RequestTLSGDAndTransformToOffset16;
    case ELF::R_MIPS_TLS_LDM:
      return RequestTLSLDMAndTransformToOffset16;
    case ELF::R_MIPS_TLS_DTPREL_HI16:
      return DTPRelHi16;
    case ELF::R_MIPS_TLS_DTPREL_LO16:
      return DTPRelLo16;
    case ELF::R_MIPS_TLS_DTPREL32:
      return DTPRel32;
    case ELF::R_MIPS_TLS_DTPREL64:
      return DTPRel64;
    case ELF::R_MIPS_TLS_GOTTPREL:
    case ELF::R_MIPS_TLS_TPREL32:
    case ELF::R_MIPS_TLS_TPREL64:
    case ELF::R_MIPS_TLS_TPREL_HI16:
    case ELF::R_MIPS_TLS_TPREL_LO16:
      return make_error<JITLinkError>(
          "initial/local-exec MIPS TLS relocations are unsupported");
    default:
      return make_error<JITLinkError>("unsupported MIPS relocation " +
                                      Twine(object::getELFRelocationTypeName(
                                          ELF::EM_MIPS, R.Types.front())));
    }
  }

  Error addRelocations() override {
    if (auto Err = validateSymbols())
      return Err;
    for (const auto &RelSect : Base::Sections) {
      if (RelSect.sh_type != ELF::SHT_REL && RelSect.sh_type != ELF::SHT_RELA)
        continue;
      if ((ABI != MipsABI::O32) != (RelSect.sh_type == ELF::SHT_RELA))
        return make_error<JITLinkError>(
            Base::G->getName() +
            ": invalid MIPS relocation-section format for ABI");
      auto FixupSect = Base::Obj.getSection(RelSect.sh_info);
      if (!FixupSect)
        return FixupSect.takeError();
      Block *B = Base::getGraphBlock(RelSect.sh_info);
      if (!B)
        continue;
      SmallVector<Reloc> Rs;
      if (auto Err = collectRelocs(RelSect, *B, Rs))
        return Err;
      for (const Reloc &R : Rs) {
        uint8_t Primary = R.Types.front();
        if (Primary == ELF::R_MIPS_NONE || Primary == ELF::R_MIPS_JALR)
          continue;
        Symbol *Target = Base::getGraphSymbol(R.Symbol);
        if (!Target)
          return error(**FixupSect, R.Offset,
                       "relocation references missing symbol index " +
                           Twine(R.Symbol));
        auto Kind = edgeKind(R, *Target);
        if (!Kind)
          return joinErrors(
              error(**FixupSect, R.Offset, "cannot lower relocation"),
              Kind.takeError());
        auto FixupAddress = orc::ExecutorAddr((*FixupSect)->sh_addr) + R.Offset;
        Edge::OffsetT Offset = FixupAddress - B->getAddress();
        unsigned FixupSize = getFixupSize(*Kind);
        if (Offset > B->getSize() || FixupSize > B->getSize() - Offset)
          return error(**FixupSect, R.Offset,
                       "relocation fixup extends past the end of its block");
        B->addEdge(*Kind, Offset, *Target, R.Addend);
      }
    }
    return Error::success();
  }
};

static Symbol &getOrCreateTLSBase(LinkGraph &G) {
  auto Name = G.intern(TLSBaseName);
  if (auto *S = G.findDefinedSymbolByName(Name))
    return *S;
  Section *TLS = G.findSectionByName(".tdata");
  if (!TLS || TLS->empty())
    TLS = G.findSectionByName(".tbss");
  if (!TLS || TLS->empty()) {
    auto &S =
        G.createSection(".tdata", orc::MemProt::Read | orc::MemProt::Write);
    auto &B = G.createMutableContentBlock(S, 0, orc::ExecutorAddr(), 1, 0);
    return G.addDefinedSymbol(B, 0, Name, 0, Linkage::Strong, Scope::Local,
                              false, true);
  }
  auto &B = **TLS->blocks().begin();
  return G.addDefinedSymbol(B, 0, Name, 0, Linkage::Strong, Scope::Local, false,
                            true);
}

class MipsTableManager {
public:
  explicit MipsTableManager(LinkGraph &G) : G(G) {}

  bool visitEdge(LinkGraph &, Block *, Edge &E) {
    switch (E.getKind()) {
    case RequestGOTAndTransformToOffset16:
      rewriteGOT(E, GOTEntryKind::Exact, GOTOffset16);
      return true;
    case RequestGOTPageAndTransformToOffset16:
      rewriteGOT(E, GOTEntryKind::Page, GOTOffset16);
      return true;
    case RequestGOTAndTransformToOffsetHi16:
      rewriteGOT(E, GOTEntryKind::Exact, GOTOffsetHi16);
      return true;
    case RequestGOTAndTransformToOffsetLo16:
      rewriteGOT(E, GOTEntryKind::Exact, GOTOffsetLo16);
      return true;
    case RequestTLSGDAndTransformToOffset16:
      rewriteTLS(E, TLSDescriptorKind::GeneralDynamic);
      return true;
    case RequestTLSLDMAndTransformToOffset16:
      rewriteTLS(E, TLSDescriptorKind::LocalDynamic);
      return true;
    case Jump26:
      if (!E.getTarget().isDefined()) {
        rewriteBranchToStub(E);
        return true;
      }
      return false;
    case PC26S2:
      if (mips::isR6(G) && !E.getTarget().isDefined()) {
        rewriteBranchToStub(E);
        return true;
      }
      return false;
    default:
      return false;
    }
  }

private:
  enum class GOTEntryKind { Exact, Page };
  enum class TLSDescriptorKind { GeneralDynamic, LocalDynamic };

  using GOTKey = std::tuple<Symbol *, int64_t, GOTEntryKind>;
  using StubKey = std::tuple<Symbol *, int64_t>;
  LinkGraph &G;
  DenseMap<GOTKey, Symbol *> GOTEntries;
  DenseMap<StubKey, Symbol *> Stubs;
  DenseMap<std::pair<Symbol *, int64_t>, Symbol *> TLSGDEntries;
  Symbol *TLSLDMEntry = nullptr;

  Section &getSection(StringRef Name, orc::MemProt Prot) {
    if (auto *S = G.findSectionByName(Name))
      return *S;
    return G.createSection(Name, Prot);
  }

  Symbol &getGOT(Symbol &Target, int64_t Addend, GOTEntryKind Kind) {
    GOTKey Key{&Target, Addend, Kind};
    auto I = GOTEntries.find(Key);
    if (I != GOTEntries.end())
      return *I->second;
    auto &Sec =
        getSection(GOTSectionName, orc::MemProt::Read | orc::MemProt::Write);
    auto &Entry = mips::createAnonymousPointer(G, Sec);
    Edge::Kind PointerKind = Kind == GOTEntryKind::Page
                                 ? getPagePointerEdgeKind(G)
                                 : mips::getPointerEdgeKind(G);
    Entry.getBlock().addEdge(PointerKind, 0, Target, Addend);
    GOTEntries[Key] = &Entry;
    return Entry;
  }

  void rewriteGOT(Edge &E, GOTEntryKind EntryKind, Edge::Kind NewKind) {
    Symbol &Entry = getGOT(E.getTarget(), E.getAddend(), EntryKind);
    E.setKind(NewKind);
    E.setTarget(Entry);
    E.setAddend(0);
  }

  Symbol &createTLSInfo(Symbol &Target, int64_t Addend) {
    auto &Sec = getSection(TLSInfoSectionName,
                           orc::MemProt::Read | orc::MemProt::Write);
    auto &B =
        G.createMutableContentBlock(Sec, 2 * G.getPointerSize(),
                                    orc::ExecutorAddr(), G.getPointerSize(), 0);
    B.addEdge(mips::getPointerEdgeKind(G), G.getPointerSize(), Target, Addend);
    return G.addAnonymousSymbol(B, 0, B.getSize(), false, false);
  }

  void rewriteTLS(Edge &E, TLSDescriptorKind Kind) {
    Symbol *Entry = nullptr;
    if (Kind == TLSDescriptorKind::LocalDynamic) {
      if (!TLSLDMEntry)
        TLSLDMEntry = &createTLSInfo(getOrCreateTLSBase(G), 0);
      Entry = TLSLDMEntry;
    } else {
      auto Key = std::make_pair(&E.getTarget(), E.getAddend());
      auto I = TLSGDEntries.find(Key);
      if (I == TLSGDEntries.end())
        I = TLSGDEntries
                .try_emplace(Key, &createTLSInfo(E.getTarget(), E.getAddend()))
                .first;
      Entry = I->second;
    }
    E.setKind(GOTOffset16);
    E.setTarget(*Entry);
    E.setAddend(0);
  }

  void rewriteBranchToStub(Edge &E) {
    // PC26_S2 uses PC + 4; JITLink edges use the fixup address.
    int64_t PointerAddend = E.getAddend();
    int64_t BranchAddend = 0;
    if (E.getKind() == PC26S2) {
      PointerAddend += CompactBranchPCBias;
      BranchAddend = -CompactBranchPCBias;
    }

    StubKey Key{&E.getTarget(), PointerAddend};
    auto I = Stubs.find(Key);
    if (I == Stubs.end()) {
      auto &Ptr = getGOT(E.getTarget(), PointerAddend, GOTEntryKind::Exact);
      auto &Sec =
          getSection(StubsSectionName, orc::MemProt::Read | orc::MemProt::Exec);
      I = Stubs
              .try_emplace(Key,
                           &mips::createAnonymousPointerJumpStub(G, Sec, Ptr))
              .first;
    }
    E.setTarget(*I->second);
    E.setAddend(BranchAddend);
  }
};

static Error buildTables(LinkGraph &G) {
  MipsTableManager Tables(G);
  visitExistingEdges(G, Tables);
  return Error::success();
}

static Symbol &defineMagic(LinkGraph &G, Block &Anchor, StringRef Name,
                           uint64_t Offset) {
  auto N = G.intern(Name);
  if (auto *S = G.findDefinedSymbolByName(N)) {
    S->setLive(true);
    return *S;
  }
  if (auto *S = G.findExternalSymbolByName(N)) {
    G.makeDefined(*S, Anchor, Offset, 0, Linkage::Strong, Scope::Local, true);
    return *S;
  }
  return G.addDefinedSymbol(Anchor, Offset, N, 0, Linkage::Strong, Scope::Local,
                            false, true);
}

static void prepareGPRegion(LinkGraph &G,
                            ArrayRef<std::string> GPRelSectionNames) {
  Section *GOT = G.findSectionByName(GOTSectionName);
  if (!GOT)
    GOT = &G.createSection(GOTSectionName,
                           orc::MemProt::Read | orc::MemProt::Write);
  // Keep the reserved GP window as content so it can share the GOT section.
  auto &Anchor = G.createMutableContentBlock(
      *GOT, GPAnchorOffset, orc::ExecutorAddr(), G.getPointerSize(), 0);
  defineMagic(G, Anchor, GOTSymbolName, 0);
  defineMagic(G, Anchor, GPName, GPAnchorOffset);
  defineMagic(G, Anchor, GPDispName, GPAnchorOffset);
  defineMagic(G, Anchor, LocalGPName, GPAnchorOffset);

  for (const std::string &Name : GPRelSectionNames)
    if (auto *S = G.findSectionByName(Name)) {
      SmallVector<Block *> Blocks(S->blocks().begin(), S->blocks().end());
      for (auto *B : Blocks)
        G.transferBlock(*B, *GOT);
    }
}

static Error orderGPRegion(LinkGraph &G) {
  auto *GOT = G.findSectionByName(GOTSectionName);
  if (!GOT)
    return Error::success();

  if (auto *TLSInfo = G.findSectionByName(TLSInfoSectionName))
    G.mergeSections(*GOT, *TLSInfo);

  auto *GOTBase = G.findDefinedSymbolByName(G.intern(GOTSymbolName));
  assert(GOTBase && &GOTBase->getBlock().getSection() == GOT &&
         GOTBase->getOffset() == 0 && "invalid MIPS GOT base");

  // BasicLayout orders equal-address blocks by size. Give synthesized entries
  // the anchor's end address to keep the reserved GP window first.
  Block &Anchor = GOTBase->getBlock();
  Anchor.setAddress(orc::ExecutorAddr());
  for (auto *B : GOT->blocks())
    if (B != &Anchor)
      B->setAddress(orc::ExecutorAddr(Anchor.getSize()));

  return Error::success();
}

template <typename ELFT>
Error validateABIFlags(const object::ELFFile<ELFT> &Obj, MipsABI ABI,
                       StringRef FileName) {
  auto Sections = Obj.sections();
  if (!Sections)
    return Sections.takeError();
  const typename ELFT::Shdr *ABIFlagsSec = nullptr;
  for (const auto &Sec : *Sections)
    if (Sec.sh_type == ELF::SHT_MIPS_ABIFLAGS) {
      if (ABIFlagsSec)
        return make_error<JITLinkError>(FileName.str() +
                                        ": multiple .MIPS.abiflags sections");
      ABIFlagsSec = &Sec;
    }
  if (!ABIFlagsSec)
    return Error::success();
  auto Contents = Obj.template getSectionContentsAsArray<char>(*ABIFlagsSec);
  if (!Contents)
    return Contents.takeError();
  if (Contents->size() != sizeof(object::Elf_Mips_ABIFlags<ELFT>))
    return make_error<JITLinkError>(FileName.str() +
                                    ": invalid .MIPS.abiflags size");
  const auto &AF = *reinterpret_cast<const object::Elf_Mips_ABIFlags<ELFT> *>(
      Contents->data());
  if (AF.version != 0)
    return make_error<JITLinkError>(FileName.str() +
                                    ": unsupported .MIPS.abiflags version");
  if ((ABI == MipsABI::N32 || ABI == MipsABI::N64) &&
      AF.gpr_size != Mips::AFL_REG_64)
    return make_error<JITLinkError>(
        FileName.str() + ": N32/N64 requires 64-bit general registers");
  if (static_cast<uint32_t>(AF.ases) &
      (Mips::AFL_ASE_MIPS16 | Mips::AFL_ASE_MICROMIPS))
    return make_error<JITLinkError>(
        FileName.str() +
        ": MIPS16 and microMIPS ABI flags are unsupported by JITLink");
  return Error::success();
}

template <typename ELFT>
Expected<std::unique_ptr<LinkGraph>>
buildGraph(object::ELFObjectFile<ELFT> &ObjFile,
           std::shared_ptr<orc::SymbolStringPool> SSP, MipsABI ABI, Triple TT,
           SubtargetFeatures Features) {
  auto &Obj = ObjFile.getELFFile();
  if (auto Err = validateABIFlags(Obj, ABI, ObjFile.getFileName()))
    return std::move(Err);
  ELFLinkGraphBuilder_mips<ELFT> Builder(ObjFile.getFileName(), Obj,
                                         std::move(SSP), std::move(TT),
                                         std::move(Features), ABI);
  auto G = Builder.buildGraph();
  if (!G)
    return G.takeError();

  SmallVector<std::string> GPRelSections;
  auto Sections = Obj.sections();
  if (!Sections)
    return Sections.takeError();
  for (const auto &Sec : *Sections)
    if (Sec.sh_flags & ELF::SHF_MIPS_GPREL) {
      auto Name = Obj.getSectionName(Sec);
      if (!Name)
        return Name.takeError();
      GPRelSections.push_back(Name->str());
    }
  prepareGPRegion(**G, GPRelSections);
  return G;
}

static Expected<MipsABI> validateHeader(const object::ELFObjectFileBase &Obj,
                                        uint8_t ELFClass, StringRef FileName) {
  uint32_t Flags = Obj.getPlatformFlags();
  if (Flags & (ELF::EF_MIPS_ARCH_ASE_M16 | ELF::EF_MIPS_MICROMIPS))
    return make_error<JITLinkError>(
        FileName.str() +
        ": MIPS16 and microMIPS objects are unsupported by JITLink");

  uint32_t ABIFlag = Flags & ELF::EF_MIPS_ABI;
  if (ELFClass == ELF::ELFCLASS64) {
    if ((Flags & ELF::EF_MIPS_ABI2) || ABIFlag != 0)
      return make_error<JITLinkError>(
          FileName.str() + ": inconsistent MIPS N64 ELF class/ABI flags");
    return MipsABI::N64;
  }
  if (Flags & ELF::EF_MIPS_ABI2) {
    if (ABIFlag != 0)
      return make_error<JITLinkError>(
          FileName.str() + ": inconsistent MIPS N32 ELF class/ABI flags");
    return MipsABI::N32;
  }
  if (ABIFlag != 0 && ABIFlag != ELF::EF_MIPS_ABI_O32)
    return make_error<JITLinkError>(FileName.str() +
                                    ": unsupported 32-bit MIPS ABI flags");
  return MipsABI::O32;
}

} // namespace

namespace llvm {
namespace jitlink {

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_mips(MemoryBufferRef ObjectBuffer,
                                  std::shared_ptr<orc::SymbolStringPool> SSP) {
  StringRef FileName = ObjectBuffer.getBufferIdentifier();
  auto ELFObj = object::ObjectFile::createELFObjectFile(ObjectBuffer);
  if (!ELFObj)
    return ELFObj.takeError();
  auto &Base = cast<object::ELFObjectFileBase>(**ELFObj);
  assert(Base.getEMachine() == ELF::EM_MIPS && "expected an ELF/MIPS object");

  StringRef Buf = ObjectBuffer.getBuffer();
  uint8_t Class = Buf[ELF::EI_CLASS];
  uint8_t Data = Buf[ELF::EI_DATA];
  auto ABI = validateHeader(Base, Class, FileName);
  if (!ABI)
    return ABI.takeError();
  auto Features = Base.getFeatures();
  if (!Features)
    return Features.takeError();

  Triple TT = Base.makeTriple();
  if (*ABI == MipsABI::N32) {
    TT.setArch(Data == ELF::ELFDATA2LSB ? Triple::mips64el : Triple::mips64);
    TT.setEnvironment(Triple::GNUABIN32);
  }

  if (Class == ELF::ELFCLASS32 && Data == ELF::ELFDATA2LSB)
    return buildGraph(cast<object::ELFObjectFile<object::ELF32LE>>(**ELFObj),
                      std::move(SSP), *ABI, TT, std::move(*Features));
  if (Class == ELF::ELFCLASS32 && Data == ELF::ELFDATA2MSB)
    return buildGraph(cast<object::ELFObjectFile<object::ELF32BE>>(**ELFObj),
                      std::move(SSP), *ABI, TT, std::move(*Features));
  if (Class == ELF::ELFCLASS64 && Data == ELF::ELFDATA2LSB)
    return buildGraph(cast<object::ELFObjectFile<object::ELF64LE>>(**ELFObj),
                      std::move(SSP), *ABI, TT, std::move(*Features));
  if (Class == ELF::ELFCLASS64 && Data == ELF::ELFDATA2MSB)
    return buildGraph(cast<object::ELFObjectFile<object::ELF64BE>>(**ELFObj),
                      std::move(SSP), *ABI, TT, std::move(*Features));
  llvm_unreachable("invalid ELF/MIPS class or byte order");
}

void link_ELF_mips(std::unique_ptr<LinkGraph> G,
                   std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  const Triple &TT = G->getTargetTriple();
  if (Ctx->shouldAddDefaultTargetPasses(TT)) {
    Config.PrePrunePasses.push_back(DWARFRecordSectionSplitter(".eh_frame"));
    Config.PrePrunePasses.push_back(
        EHFrameEdgeFixer(".eh_frame", G->getPointerSize(), Pointer32, Pointer64,
                         Delta32, Delta64, NegDelta32));
    Config.PrePrunePasses.push_back(EHFrameNullTerminator(".eh_frame"));
    if (auto MarkLive = Ctx->getMarkLivePass(TT))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);
    Config.PostPrunePasses.push_back(buildTables);
  }
  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));
  if (Ctx->shouldAddDefaultTargetPasses(TT))
    Config.PostPrunePasses.push_back(orderGPRegion);
  ELFJITLinker_mips::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace jitlink
} // namespace llvm
