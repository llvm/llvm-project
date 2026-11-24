//===-- mips.h - Generic JITLink MIPS edge kinds and utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing MIPS objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_MIPS_H
#define LLVM_EXECUTIONENGINE_JITLINK_MIPS_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {
namespace jitlink {
namespace mips {

/// MIPS fixup and table-request edge kinds.
enum EdgeKind_mips : Edge::Kind {
  /// Fixup <- Target + Addend, checked as an unsigned 32-bit pointer.
  Pointer32 = Edge::FirstRelocation,
  /// Fixup <- Target + Addend : uint64.
  Pointer64,
  /// Pointer containing the rounded 64-KiB page of Target + Addend.
  PagePointer32,
  PagePointer64,
  /// Fixup <- Target - Fixup + Addend : int32.
  Delta32,
  /// Fixup <- Target - Fixup + Addend : int64.
  Delta64,
  /// Fixup <- Fixup - Target + Addend : int32 (used by .eh_frame).
  NegDelta32,

  /// Fixup[15:0] <- Target + Addend : int16.
  Abs16,
  /// Fixup[15:0] <- (Target + Addend + 0x8000) >> 16.
  Hi16,
  /// Fixup[15:0] <- Target + Addend.
  Lo16,
  /// Fixup[15:0] <- (Target + Addend + 0x80008000) >> 32.
  Higher16,
  /// Fixup[15:0] <- (Target + Addend + 0x800080008000) >> 48.
  Highest16,

  /// J/JAL target. The target must be aligned and in the same 256-MiB region.
  Jump26,
  /// Fixup[15:0] <- (Target - Fixup + Addend) >> 2 : int16.
  PC16,
  /// Fixup <- Target - Fixup + Addend : int32.
  PC32,
  /// Release-6 compact branch immediate fields.
  PC18S3,
  PC19S2,
  PC21S2,
  PC26S2,
  /// Paired PC-relative high and low halves.
  PCHi16,
  PCLo16,
  /// _gp_disp high/low expressions. The O32 low half uses Fixup - 4 as P.
  GPDispHi16,
  GPDispLo16,

  /// Fixup <- Target + Addend - _gp.
  GPRel16,
  GPRel32,
  GPRel64,
  /// Fixup <- address of GOT entry - _gp.
  GOTOffset16,
  GOTOffsetHi16,
  GOTOffsetLo16,
  /// Low offset complementary to a rounded 64-KiB GOT page entry.
  GOTPageOffset16,

  /// Fixup <- Target + Addend - start-of-ORC-TLS-template.
  DTPRelHi16,
  DTPRelLo16,
  DTPRel32,
  DTPRel64,

  /// Compound N32/N64 %neg(%gp_rel(Target + Addend)) fixups.
  NegGPRelHi16,
  NegGPRelLo16,

  /// Create an exact-address GOT entry and rewrite to GOTOffset16.
  RequestGOTAndTransformToOffset16,
  /// Create a rounded 64-KiB page GOT entry and rewrite to GOTOffset16.
  RequestGOTPageAndTransformToOffset16,
  /// Create an exact-address GOT entry and rewrite to its high/low GP offset.
  RequestGOTAndTransformToOffsetHi16,
  RequestGOTAndTransformToOffsetLo16,
  /// Create a two-word general/local-dynamic TLS descriptor.
  RequestTLSGDAndTransformToOffset16,
  RequestTLSLDMAndTransformToOffset16,
};

LLVM_ABI const char *getEdgeKindName(Edge::Kind K);

/// Returns true if G uses the MIPS release-6 ISA.
LLVM_ABI bool isR6(const LinkGraph &G);

/// Returns Pointer32 or Pointer64, according to G's pointer ABI.
LLVM_ABI Edge::Kind getPointerEdgeKind(const LinkGraph &G);

constexpr unsigned InstructionSize = sizeof(uint32_t);
constexpr unsigned GOTPageBits = 16;
constexpr uint64_t GOTPageSize = UINT64_C(1) << GOTPageBits;
constexpr uint64_t GOTPageMask = GOTPageSize - 1;
constexpr uint64_t GOTPageBias = GOTPageSize / 2;
constexpr unsigned JumpRegionBits = 28;
constexpr uint64_t JumpRegionMask = ~maskTrailingOnes<uint64_t>(JumpRegionBits);

inline uint16_t getLo16(uint64_t Value) { return static_cast<uint16_t>(Value); }

inline uint16_t getHi16(uint64_t Value) {
  return static_cast<uint16_t>((Value + GOTPageBias) >> 16);
}

inline uint16_t getHigher16(uint64_t Value) {
  constexpr uint64_t Bias = GOTPageBias | (GOTPageBias << 16);
  return static_cast<uint16_t>((Value + Bias) >> 32);
}

inline uint16_t getHighest16(uint64_t Value) {
  constexpr uint64_t Bias =
      GOTPageBias | (GOTPageBias << 16) | (GOTPageBias << 32);
  return static_cast<uint16_t>((Value + Bias) >> 48);
}

inline uint64_t getGOTPage(uint64_t Value) {
  return (Value + GOTPageBias) & ~GOTPageMask;
}

struct PCRelEncoding {
  unsigned Bits;
  unsigned Shift;
  unsigned PCAlignment;
};

inline PCRelEncoding getPCRelEncoding(Edge::Kind Kind) {
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

inline bool needsGP(Edge::Kind K) {
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

constexpr uint32_t InstructionImm16Mask = 0x0000ffffU;
constexpr uint32_t InstructionImm26Mask = 0x03ffffffU;

inline void writeMaskedInstruction32(char *Fixup, uint32_t Mask, uint32_t Value,
                                     endianness Endianness) {
  uint32_t Instruction = support::endian::read32(Fixup, Endianness);
  support::endian::write32(Fixup, (Instruction & ~Mask) | (Value & Mask),
                           Endianness);
}

inline void writeImmediate16(char *Fixup, uint16_t Value,
                             endianness Endianness) {
  writeMaskedInstruction32(Fixup, InstructionImm16Mask, Value, Endianness);
}

inline void writeImmediate26(char *Fixup, uint32_t Value,
                             endianness Endianness) {
  writeMaskedInstruction32(Fixup, InstructionImm26Mask, Value, Endianness);
}

/// Apply fixup expression for edge to block content.
/// GPSymbol and TLSBaseSymbol supply the bases for GP-relative and DTP-relative
/// edges. They may be null when the edge does not require the respective base.
inline Error applyFixup(LinkGraph &G, Block &B, const Edge &E,
                        const Symbol *GPSymbol, const Symbol *TLSBaseSymbol) {
  char *Fixup = B.getAlreadyMutableContent().data() + E.getOffset();
  uint64_t P = B.getFixupAddress(E).getValue();
  uint64_t S = E.getTarget().getAddress().getValue();
  int64_t A = E.getAddend();
  uint64_t TargetAddress = S + A;
  const endianness Endianness = G.getEndianness();

  auto TLSBaseAddr = [&]() -> Expected<uint64_t> {
    if (TLSBaseSymbol)
      return TLSBaseSymbol->getAddress().getValue();
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
    assert(GPSymbol && "missing MIPS GP symbol");
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
    writeImmediate16(Fixup, getLo16(*GP + A - P + InstructionSize), Endianness);
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

/// Returns zero-filled pointer contents in the graph's pointer width.
LLVM_ABI ArrayRef<char> getPointerBlockContent(const LinkGraph &G);

/// Creates an anonymous pointer, optionally initialized to InitialTarget.
LLVM_ABI Symbol &createAnonymousPointer(LinkGraph &G, Section &PointerSection,
                                        Symbol *InitialTarget = nullptr,
                                        Edge::AddendT InitialAddend = 0);

/// Creates a stub that materializes PointerSymbol, loads its value into $t9,
/// and jumps to $t9. Release-6 graphs use the release-6 indirect-jump encoding.
LLVM_ABI Symbol &createAnonymousPointerJumpStub(LinkGraph &G,
                                                Section &StubSection,
                                                Symbol &PointerSymbol);

} // namespace mips
} // namespace jitlink
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_MIPS_H
