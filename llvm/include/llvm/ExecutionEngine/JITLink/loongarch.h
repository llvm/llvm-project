//= loongarch.h - Generic JITLink loongarch edge kinds, utilities -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing loongarch objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_LOONGARCH_H
#define LLVM_EXECUTIONENGINE_JITLINK_LOONGARCH_H

#include "TableManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Shared/MemoryFlags.h"
#include "llvm/Support/LEB128.h"

namespace llvm {
namespace jitlink {
namespace loongarch {

/// Represents loongarch fixups.
enum EdgeKind_loongarch : Edge::Kind {
  /// A plain 64-bit pointer value relocation.
  ///
  /// Fixup expression:
  ///   Fixup <- Target + Addend : uint64
  ///
  Pointer64 = Edge::FirstRelocation,

  /// A plain 32-bit pointer value relocation.
  ///
  /// Fixup expression:
  ///   Fixup <- Target + Addend : uint32
  ///
  /// Errors:
  ///   - The target must reside in the low 32-bits of the address space,
  ///     otherwise an out-of-range error will be returned.
  ///
  Pointer32,

  /// A 16-bit PC-relative branch.
  ///
  /// Represents a PC-relative branch to a target within +/-128Kb. The target
  /// must be 4-byte aligned.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) >> 2 : int16
  ///
  /// Notes:
  ///   The '16' in the name refers to the number operand bits and follows the
  /// naming convention used by the corresponding ELF relocations. Since the low
  /// two bits must be zero (because of the 4-byte alignment of the target) the
  /// operand is effectively a signed 18-bit number.
  ///
  /// Errors:
  ///   - The result of the unshifted part of the fixup expression must be
  ///     4-byte aligned otherwise an alignment error will be returned.
  ///   - The result of the fixup expression must fit into an int16 otherwise an
  ///     out-of-range error will be returned.
  ///
  Branch16PCRel,

  /// A 21-bit PC-relative branch.
  ///
  /// Represents a PC-relative branch to a target within +/-4Mb. The Target must
  /// be 4-byte aligned.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) >> 2 : int21
  ///
  /// Notes:
  ///   The '21' in the name refers to the number operand bits and follows the
  /// naming convention used by the corresponding ELF relocations. Since the low
  /// two bits must be zero (because of the 4-byte alignment of the target) the
  /// operand is effectively a signed 23-bit number.
  ///
  /// Errors:
  ///   - The result of the unshifted part of the fixup expression must be
  ///     4-byte aligned otherwise an alignment error will be returned.
  ///   - The result of the fixup expression must fit into an int21 otherwise an
  ///     out-of-range error will be returned.
  ///
  Branch21PCRel,

  /// A 26-bit PC-relative branch.
  ///
  /// Represents a PC-relative call or branch to a target within +/-128Mb. The
  /// target must be 4-byte aligned.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) >> 2 : int26
  ///
  /// Notes:
  ///   The '26' in the name refers to the number operand bits and follows the
  /// naming convention used by the corresponding ELF relocations. Since the low
  /// two bits must be zero (because of the 4-byte alignment of the target) the
  /// operand is effectively a signed 28-bit number.
  ///
  /// Errors:
  ///   - The result of the unshifted part of the fixup expression must be
  ///     4-byte aligned otherwise an alignment error will be returned.
  ///   - The result of the fixup expression must fit into an int26 otherwise an
  ///     out-of-range error will be returned.
  ///
  Branch26PCRel,

  /// A 32-bit delta.
  ///
  /// Delta from the fixup to the target.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta32,

  /// A 32-bit negative delta.
  ///
  /// Delta from the target back to the fixup.
  ///
  /// Fixup expression:
  ///   Fixup <- Fixup - Target + Addend : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  NegDelta32,

  /// A 64-bit delta.
  ///
  /// Delta from the fixup to the target.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend : int64
  ///
  Delta64,

  /// The signed 20-bit delta from the fixup page to the page containing the
  /// target.
  ///
  /// Fixup expression:
  ///   Fixup <- (((Target + Addend + ((Target + Addend) & 0x800)) & ~0xfff)
  //              - (Fixup & ~0xfff)) >> 12 : int20
  ///
  /// Notes:
  ///   For PCALAU12I fixups.
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int20 otherwise an
  ///     out-of-range error will be returned.
  ///
  Page20,

  /// The 12-bit offset of the target within its page.
  ///
  /// Typically used to fix up ADDI/LD_W/LD_D immediates.
  ///
  /// Fixup expression:
  ///   Fixup <- ((Target + Addend) >> Shift) & 0xfff : int12
  ///
  PageOffset12,

  /// A GOT entry getter/constructor, transformed to Page20 pointing at the GOT
  /// entry for the original target.
  ///
  /// Indicates that this edge should be transformed into a Page20 targeting
  /// the GOT entry for the edge's current target, maintaining the same addend.
  /// A GOT entry for the target should be created if one does not already
  /// exist.
  ///
  /// Edges of this kind are usually handled by a GOT/PLT builder pass inserted
  /// by default.
  ///
  /// Fixup expression:
  ///   NONE
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to handle edges of this kind prior to the fixup
  ///     phase will result in an assert/unreachable during the fixup phase.
  ///
  RequestGOTAndTransformToPage20,

  /// A GOT entry getter/constructor, transformed to Pageoffset12 pointing at
  /// the GOT entry for the original target.
  ///
  /// Indicates that this edge should be transformed into a PageOffset12
  /// targeting the GOT entry for the edge's current target, maintaining the
  /// same addend. A GOT entry for the target should be created if one does not
  /// already exist.
  ///
  /// Edges of this kind are usually handled by a GOT/PLT builder pass inserted
  /// by default.
  ///
  /// Fixup expression:
  ///   NONE
  ///
  RequestGOTAndTransformToPageOffset12,

  /// A 36-bit PC-relative call.
  ///
  /// Represents a PC-relative call to a target within [-128G - 0x20000, +128G
  /// - 0x20000). The target must be 4-byte aligned. For adjacent pcaddu18i+jirl
  /// instruction pairs.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) >> 2 : int36
  ///
  /// Notes:
  ///   The '36' in the name refers to the number operand bits and follows the
  /// naming convention used by the corresponding ELF relocations. Since the low
  /// two bits must be zero (because of the 4-byte alignment of the target) the
  /// operand is effectively a signed 38-bit number.
  ///
  /// Errors:
  ///   - The result of the unshifted part of the fixup expression must be
  ///     4-byte aligned otherwise an alignment error will be returned.
  ///   - The result of the fixup expression must fit into an int36 otherwise an
  ///     out-of-range error will be returned.
  ///
  Call36PCRel,

  /// low 6 bits label addition
  ///
  /// Fixup expression:
  ///   Fixup <- (*{1}Fixup + (Target + Addend) & 0x3f) : int8
  ///
  Add6,

  /// 8 bits label addition
  ///
  /// Fixup expression:
  ///   Fixup <- (*{1}Fixup + Target + Addend) : int8
  ///
  Add8,

  /// 16 bits label addition
  ///
  /// Fixup expression:
  ///   Fixup <- (*{2}Fixup + Target + Addend) : int16
  ///
  Add16,

  /// 32 bits label addition
  ///
  /// Fixup expression:
  ///   Fixup <- (*{4}Fixup + Target + Addend) : int32
  ///
  Add32,

  /// 64 bits label addition
  ///
  /// Fixup expression:
  ///   Fixup <- (*{8}Fixup + Target + Addend) : int64
  ///
  Add64,

  /// ULEB128 bits label addition
  ///
  /// Fixup expression:
  ///   Fixup <- (Fixup + Target + Addend) : uleb128
  ///
  AddUleb128,

  /// low 6 bits label subtraction
  ///
  /// Fixup expression:
  ///   Fixup <- (*{1}Fixup - (Target + Addend) & 0x3f) : int8
  ///
  Sub6,

  /// 8 bits label subtraction
  ///
  /// Fixup expression:
  ///   Fixup <- (*{1}Fixup - Target - Addend) : int8
  ///
  Sub8,

  /// 16 bits label subtraction
  ///
  /// Fixup expression:
  ///   Fixup <- (*{2}Fixup - Target - Addend) : int16
  ///
  Sub16,

  /// 32 bits label subtraction
  ///
  /// Fixup expression:
  ///   Fixup <- (*{4}Fixup - Target - Addend) : int32
  ///
  Sub32,

  /// 64 bits label subtraction
  ///
  /// Fixup expression:
  ///   Fixup <- (*{8}Fixup - Target - Addend) : int64
  ///
  Sub64,

  /// ULEB128 bits label subtraction
  ///
  /// Fixup expression:
  ///   Fixup <- (Fixup - Target - Addend) : uleb128
  ///
  SubUleb128,

  /// Alignment requirement used by linker relaxation.
  ///
  /// Linker relaxation will use this to ensure all code sequences are properly
  /// aligned and then remove these edges from the graph.
  ///
  AlignRelaxable,
};

/// Returns a string name for the given loongarch edge. For debugging purposes
/// only.
const char *getEdgeKindName(Edge::Kind K);

// Returns extract bits Val[Hi:Lo].
inline uint32_t extractBits(uint64_t Val, unsigned Hi, unsigned Lo) {
  return Hi == 63 ? Val >> Lo : (Val & ((((uint64_t)1 << (Hi + 1)) - 1))) >> Lo;
}

/// Apply fixup expression for edge to block content.
inline Error applyFixup(LinkGraph &G, Block &B, const Edge &E) {
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
    int64_t Value =
        TargetAddress + *(reinterpret_cast<const int8_t *>(FixupPtr)) + Addend;
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
    uint64_t Orig = decodeULEB128((reinterpret_cast<const uint8_t *>(FixupPtr)),
                                  &Count, nullptr, &Error);

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
    int64_t Value =
        *(reinterpret_cast<const int8_t *>(FixupPtr)) - TargetAddress - Addend;
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
    uint64_t Orig = decodeULEB128((reinterpret_cast<const uint8_t *>(FixupPtr)),
                                  &Count, nullptr, &Error);

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

/// loongarch null pointer content.
extern const char NullPointerContent[8];
inline ArrayRef<char> getGOTEntryBlockContent(LinkGraph &G) {
  return {reinterpret_cast<const char *>(NullPointerContent),
          G.getPointerSize()};
}

/// loongarch stub content.
///
/// Contains the instruction sequence for an indirect jump via an in-memory
/// pointer:
///   pcalau12i $t8, %page20(ptr)
///   ld.[w/d]  $t8, %pageoff12(ptr)
///   jr        $t8
constexpr size_t StubEntrySize = 12;
extern const uint8_t LA64StubContent[StubEntrySize];
extern const uint8_t LA32StubContent[StubEntrySize];
inline ArrayRef<char> getStubBlockContent(LinkGraph &G) {
  auto StubContent =
      G.getPointerSize() == 8 ? LA64StubContent : LA32StubContent;
  return {reinterpret_cast<const char *>(StubContent), StubEntrySize};
}

/// Creates a new pointer block in the given section and returns an
/// Anonymous symbol pointing to it.
///
/// If InitialTarget is given then an Pointer64 relocation will be added to the
/// block pointing at InitialTarget.
///
/// The pointer block will have the following default values:
///   alignment: PointerSize
///   alignment-offset: 0
inline Symbol &createAnonymousPointer(LinkGraph &G, Section &PointerSection,
                                      Symbol *InitialTarget = nullptr,
                                      uint64_t InitialAddend = 0) {
  auto &B = G.createContentBlock(PointerSection, getGOTEntryBlockContent(G),
                                 orc::ExecutorAddr(), G.getPointerSize(), 0);
  if (InitialTarget)
    B.addEdge(G.getPointerSize() == 8 ? Pointer64 : Pointer32, 0,
              *InitialTarget, InitialAddend);
  return G.addAnonymousSymbol(B, 0, G.getPointerSize(), false, false);
}

/// Create a jump stub that jumps via the pointer at the given symbol and
/// an anonymous symbol pointing to it. Return the anonymous symbol.
inline Symbol &createAnonymousPointerJumpStub(LinkGraph &G,
                                              Section &StubSection,
                                              Symbol &PointerSymbol) {
  Block &StubContentBlock = G.createContentBlock(
      StubSection, getStubBlockContent(G), orc::ExecutorAddr(), 4, 0);
  StubContentBlock.addEdge(Page20, 0, PointerSymbol, 0);
  StubContentBlock.addEdge(PageOffset12, 4, PointerSymbol, 0);
  return G.addAnonymousSymbol(StubContentBlock, 0, StubEntrySize, true, false);
}

/// Global Offset Table Builder.
class GOTTableManager : public TableManager<GOTTableManager> {
public:
  static StringRef getSectionName() { return "$__GOT"; }

  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    Edge::Kind KindToSet = Edge::Invalid;
    switch (E.getKind()) {
    case RequestGOTAndTransformToPage20:
      KindToSet = Page20;
      break;
    case RequestGOTAndTransformToPageOffset12:
      KindToSet = PageOffset12;
      break;
    default:
      return false;
    }
    assert(KindToSet != Edge::Invalid &&
           "Fell through switch, but no new kind to set");
    DEBUG_WITH_TYPE("jitlink", {
      dbgs() << "  Fixing " << G.getEdgeKindName(E.getKind()) << " edge at "
             << B->getFixupAddress(E) << " (" << B->getAddress() << " + "
             << formatv("{0:x}", E.getOffset()) << ")\n";
    });
    E.setKind(KindToSet);
    E.setTarget(getEntryForTarget(G, E.getTarget()));
    return true;
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Target) {
    return createAnonymousPointer(G, getGOTSection(G), &Target);
  }

private:
  Section &getGOTSection(LinkGraph &G) {
    if (!GOTSection)
      GOTSection = &G.createSection(getSectionName(),
                                    orc::MemProt::Read | orc::MemProt::Exec);
    return *GOTSection;
  }

  Section *GOTSection = nullptr;
};

/// Procedure Linkage Table Builder.
class PLTTableManager : public TableManager<PLTTableManager> {
public:
  PLTTableManager(GOTTableManager &GOT) : GOT(GOT) {}

  static StringRef getSectionName() { return "$__STUBS"; }

  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    if ((E.getKind() == Branch26PCRel || E.getKind() == Call36PCRel) &&
        !E.getTarget().isDefined()) {
      DEBUG_WITH_TYPE("jitlink", {
        dbgs() << "  Fixing " << G.getEdgeKindName(E.getKind()) << " edge at "
               << B->getFixupAddress(E) << " (" << B->getAddress() << " + "
               << formatv("{0:x}", E.getOffset()) << ")\n";
      });
      E.setTarget(getEntryForTarget(G, E.getTarget()));
      return true;
    }
    return false;
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Target) {
    return createAnonymousPointerJumpStub(G, getStubsSection(G),
                                          GOT.getEntryForTarget(G, Target));
  }

public:
  Section &getStubsSection(LinkGraph &G) {
    if (!StubsSection)
      StubsSection = &G.createSection(getSectionName(),
                                      orc::MemProt::Read | orc::MemProt::Exec);
    return *StubsSection;
  }

  GOTTableManager &GOT;
  Section *StubsSection = nullptr;
};

} // namespace loongarch
} // namespace jitlink
} // namespace llvm

#endif
