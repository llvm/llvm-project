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
