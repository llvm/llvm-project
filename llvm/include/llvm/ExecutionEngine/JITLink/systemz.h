//=== systemz.h - Generic JITLink systemz edge kinds, utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing systemz objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_SYSTEMZ_H
#define LLVM_EXECUTIONENGINE_JITLINK_SYSTEMZ_H

#include "TableManager.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"

using namespace llvm::support::endian;

namespace llvm {
namespace jitlink {
namespace systemz {

/// Represents systemz fixups and other systemz-specific edge kinds.
enum EdgeKind_systemz : Edge::Kind {

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

  /// A plain 20-bit pointer value relocation.
  ///
  /// Fixup expression:
  ///   Fixup <- Target + Addend : uint20
  ///
  /// Errors:
  ///   - The target must reside in the mid 20-bits of the address space,
  ///     otherwise an out-of-range error will be returned.
  ///
  Pointer20,

  /// A plain 16-bit pointer value relocation.
  ///
  /// Fixup expression:
  ///   Fixup <- Target + Addend : uint16
  ///
  /// Errors:
  ///   - The target must reside in the low 16-bits of the address space,
  ///     otherwise an out-of-range error will be returned.
  ///
  Pointer16,

  /// A plain 12-bit pointer value relocation.
  ///
  /// Fixup expression:
  ///   Fixup <- Target + Addend : uint12
  ///
  /// Errors:
  ///   - The target must reside in the low 12-bits of the address space,
  ///     otherwise an out-of-range error will be returned.
  ///
  Pointer12,

  /// A plain 8-bit pointer value relocation.
  ///
  /// Fixup expression:
  ///   Fixup <- Target + Addend : uint8
  ///
  /// Errors:
  ///   - The target must reside in the low 8-bits of the address space,
  ///     otherwise an out-of-range error will be returned.
  ///
  Pointer8,

  /// A 64-bit delta.
  ///
  /// Delta from the fixup to the target.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend : int64
  ///
  Delta64,

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

  /// A 16-bit delta.
  ///
  /// Delta from the fixup to the target.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend : int16
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int16, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta16,

  /// A 32-bit delta.
  ///
  /// Delta from the fixup to the target.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend : int32) >> 1
  ///
  /// Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int33, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  Delta32dbl,

  /// A 24-bit delta.
  ///
  /// Delta from the fixup to the target.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend : int24) >> 1
  ///
  /// Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int25, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  Delta24dbl,

  /// A 16-bit delta.
  ///
  /// Delta from the fixup to the target.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend : int16) >> 1
  ///
  /// Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int17, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  Delta16dbl,

  /// A 12-bit delta.
  ///
  /// Delta from the fixup to the target.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend : int12) >> 1
  ///
  /// Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int13, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  Delta12dbl,

  /// A 64-bit negative delta.
  ///
  /// Delta from target back to the fixup.
  ///
  /// Fixup expression:
  ///   Fixup <- Fixup - Target + Addend : int64
  ///
  NegDelta64,

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
  NegDelta32,

  /// A 64-bit GOT delta.
  ///
  /// Delta from the global offset table to the target
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTSymbol + Addend : int64
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  Delta64FromGOT,

  /// A 32-bit GOT delta.
  ///
  /// Delta from the global offset table to the target
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTSymbol + Addend : int32
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  Delta32FromGOT,

  /// A 16-bit GOT delta.
  ///
  /// Delta from the global offset table to the target
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTSymbol + Addend : int16
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int16, otherwise
  ///     an out-of-range error will be returned.
  Delta16FromGOT,

  /// A 32-bit PC-relative branch.
  ///
  /// Represents a PC-relative call or branch to a target. This can be used to
  /// identify, record, and/or patch call sites.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) >> 1 : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int33, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  BranchPCRelPLT32dbl,

  /// A 24-bit PC-relative branch.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) >> 1 : int24
  ///
  /// Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int25, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  BranchPCRelPLT24dbl,

  /// A 16-bit PC-relative branch.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) >> 1 : int16
  ///
  /// Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int17, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  BranchPCRelPLT16dbl,

  /// A 12-bit PC-relative branch.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) >> 1 : int12
  ///
  /// Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int13, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  BranchPCRelPLT12dbl,

  /// A 64-bit PC-relative PLT address.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend : int64
  ///
  BranchPCRelPLT64,

  /// A 32-bit PC-relative PLT address.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  BranchPCRelPLT32,

  /// A 32-bit PC-relative branch to a pointer jump stub.
  /// Create a jump stub block that jumps via the pointer at the given symbol.
  ///
  /// Stub Content:
  ///   larl %r1, ptr
  ///   lg   %r1, 0(%r1)
  ///   j     %r1
  ///
  ///  Fixup expression at offset 2 of branch Instruction:
  ///    Fixup <- (Target - Fixup + Addend) >> 1 : int32
  ///
  ///  Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int33, otherwise an out-of-range error will be returned.
  ///     an out-of-range error will be returned.
  ///
  Branch32dblToStub,

  /// A 64-bit offset from GOT to PLT.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int64
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///
  DeltaPLT64FromGOT,

  /// A 32-bit offset from GOT to PLT.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int32
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  DeltaPLT32FromGOT,

  /// A 16-bit offset from GOT to PLT.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int16
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int16, otherwise
  ///     an out-of-range error will be returned.
  ///
  DeltaPLT16FromGOT,

  /// A 64-bit GOT offset.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int64
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///
  Delta64GOT,

  /// A 32-bit GOT offset.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int32
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta32GOT,

  /// A 20-bit GOT offset.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int20
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int20, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta20GOT,

  /// A 16-bit GOT offset.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int16
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int16, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta16GOT,

  /// A 12-bit GOT offset.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int12
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int12, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta12GOT,

  /// A 32-bit PC rel. offset to GOT.
  ///
  /// Fixup expression:
  ///   Fixup <- GOTBase - Fixup + Addend : int32
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  DeltaPCRelGOT,

  /// A 32-bit PC rel. offset to GOT shifted by 1.
  ///
  /// Fixup expression:
  ///   Fixup <- (GOTBase - Fixup + Addend) >> 1 : int32
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int33, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  DeltaPCRelGOTdbl,

  /// A 64-bit offset to Jump Slot.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int64
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///
  Delta64JumpSlot,

  /// A 32-bit offset to Jump Slot.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int32
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta32JumpSlot,

  /// A 20-bit offset to Jump Slot.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int20
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int20, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta20JumpSlot,

  /// A 16-bit offset to Jump Slot.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int16
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int16, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta16JumpSlot,

  /// A 12-bit offset to Jump Slot.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int12
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int12, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta12JumpSlot,

  /// A 32-bit PC rel. offset to Jump Slot.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) >> 1 : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int33, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  PCRel32JumpSlot,

  /// A 32-bit PC rel. to GOT entry >> 1.
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) >> 1 : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression before shifting right by 1 must
  ///     fit into an int33, otherwise an out-of-range error will be returned.
  ///   - The result of the fixup expression  before shifting right by 1 must
  ///     be multiple of 2, otherwise an alignment error will be returned.
  ///
  PCRel32GOTEntry,

};

/// Returns a string name for the given systemz edge. For debugging purposes
/// only
const char *getEdgeKindName(Edge::Kind K);

/// Apply fixup expression for edge to block content.
inline Error applyFixup(LinkGraph &G, Block &B, const Edge &E,
                        const Symbol *GOTSymbol) {
  using namespace support;

  char *BlockWorkingMem = B.getAlreadyMutableContent().data();
  char *FixupPtr = BlockWorkingMem + E.getOffset();
  orc::ExecutorAddr FixupAddress = B.getAddress() + E.getOffset();
  int64_t S = E.getTarget().getAddress().getValue();
  int64_t A = E.getAddend();
  int64_t P = FixupAddress.getValue();
  int64_t GOTBase = GOTSymbol ? GOTSymbol->getAddress().getValue() : 0;
  Edge::Kind K = E.getKind();

  DEBUG_WITH_TYPE("jitlink", {
    dbgs() << "    Applying fixup on " << G.getEdgeKindName(K)
           << " edge, (S, A, P, .GOT.) = (" << formatv("{0:x}", S) << ", "
           << formatv("{0:x}", A) << ", " << formatv("{0:x}", P) << ", "
           << formatv("{0:x}", GOTBase) << ")\n";
  });

  const auto isAlignmentCorrect = [](uint64_t Value, int N) {
    return (Value & (N - 1)) ? false : true;
  };

  switch (K) {
  case Pointer64: {
    uint64_t Value = S + A;
    write64be(FixupPtr, Value);
    break;
  }
  case Pointer32: {
    uint64_t Value = S + A;
    if (!LLVM_UNLIKELY(isUInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(ubig32_t *)FixupPtr = Value;
    break;
  }
  case Pointer20: {
    uint64_t Value = S + A;
    if (!LLVM_UNLIKELY(isInt<20>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write32be(FixupPtr, (read32be(FixupPtr) & 0xF00000FF) |
                            ((Value & 0xFFF) << 16) | ((Value & 0xFF000) >> 4));
    break;
  }
  case Pointer16: {
    uint64_t Value = S + A;
    if (!LLVM_UNLIKELY(isUInt<16>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(ubig16_t *)FixupPtr = Value;
    break;
  }
  case Pointer12: {
    uint64_t Value = S + A;
    if (!LLVM_UNLIKELY(isUInt<12>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write16be(FixupPtr, (read16be(FixupPtr) & 0xF000) | Value);
    break;
  }
  case Pointer8: {
    uint64_t Value = S + A;
    if (!LLVM_UNLIKELY(isUInt<8>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(uint8_t *)FixupPtr = Value;
    break;
  }
  case Delta64: {
    int64_t Value = S + A - P;
    *(big64_t *)FixupPtr = Value;
    break;
  }
  case Delta32: {
    int64_t Value = S + A - P;
    if (!LLVM_UNLIKELY(isInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big32_t *)FixupPtr = Value;
    break;
  }
  case Delta16: {
    int64_t Value = S + A - P;
    if (!LLVM_UNLIKELY(isInt<16>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big16_t *)FixupPtr = Value;
    break;
  }
  case NegDelta32: {
    int64_t Value = P + A - S;
    if (!LLVM_UNLIKELY(isInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big32_t *)FixupPtr = Value;
    break;
  }
  case Delta32dbl: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<33>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write32be(FixupPtr, Value >> 1);
    break;
  }
  case Delta24dbl: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<25>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    FixupPtr[0] = Value >> 17;
    FixupPtr[1] = Value >> 9;
    FixupPtr[2] = Value >> 1;
    break;
  }
  case Delta16dbl: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<17>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write16be(FixupPtr, Value >> 1);
    break;
  }
  case Delta12dbl: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<13>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write16be(FixupPtr,
              (read16be(FixupPtr) & 0xF000) | ((Value >> 1) & 0x0FFF));
    break;
  }
  case Delta64FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = S - GOTBase + A;
    *(big64_t *)FixupPtr = Value;
    break;
  }
  case Delta32FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big32_t *)FixupPtr = Value;
    break;
  }
  case Delta16FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isInt<16>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big16_t *)FixupPtr = Value;
    break;
  }
  case DeltaPCRelGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = GOTBase + A - P;
    *(big32_t *)FixupPtr = Value;
    break;
  }
  case DeltaPCRelGOTdbl: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = (GOTBase + A - P);
    if (!LLVM_UNLIKELY(isInt<33>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write32be(FixupPtr, Value >> 1);
    break;
  }
  case BranchPCRelPLT32dbl: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<33>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write32be(FixupPtr, Value >> 1);
    break;
  }
  case BranchPCRelPLT24dbl: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<25>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    FixupPtr[0] = Value >> 17;
    FixupPtr[1] = Value >> 9;
    FixupPtr[2] = Value >> 1;
    break;
  }
  case BranchPCRelPLT16dbl: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<17>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write16be(FixupPtr, Value >> 1);
    break;
  }
  case BranchPCRelPLT12dbl: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<13>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write16be(FixupPtr,
              (read16be(FixupPtr) & 0xF000) | ((Value >> 1) & 0x0FFF));
    break;
  }
  case BranchPCRelPLT64: {
    int64_t Value = (S + A - P);
    *(big64_t *)FixupPtr = Value;
    break;
  }
  case BranchPCRelPLT32: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big32_t *)FixupPtr = Value;
    break;
  }
  case DeltaPLT64FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = (S + A - GOTBase);
    *(big64_t *)FixupPtr = Value;
    break;
  }
  case DeltaPLT32FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = (S + A - GOTBase);
    if (!LLVM_UNLIKELY(isInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big32_t *)FixupPtr = Value;
    break;
  }
  case DeltaPLT16FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = (S + A - GOTBase);
    if (!LLVM_UNLIKELY(isInt<16>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big16_t *)FixupPtr = Value;
    break;
  }
  case Branch32dblToStub: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<33>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    char *AddrToPatch = FixupPtr + 2;
    *(big32_t *)AddrToPatch = (Value >> 1);
    break;
  }
  case PCRel32GOTEntry: {
    int64_t Value = (S + A - P);
    if (!LLVM_UNLIKELY(isInt<33>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write32be(FixupPtr, Value >> 1);
    break;
  }
  case Delta64GOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = S - GOTBase + A;
    *(big64_t *)FixupPtr = Value;
    break;
  }
  case Delta32GOT: {
    assert(GOTSymbol && "No GOT section symbol");
    uint64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isUInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big32_t *)FixupPtr = Value;
    break;
  }
  case Delta20GOT: {
    assert(GOTSymbol && "No GOT section symbol");
    uint64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isInt<20>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write32be(FixupPtr, (read32be(FixupPtr) & 0xF00000FF) |
                            ((Value & 0xFFF) << 16) | ((Value & 0xFF000) >> 4));
    break;
  }
  case Delta16GOT: {
    assert(GOTSymbol && "No GOT section symbol");
    uint64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isUInt<16>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big16_t *)FixupPtr = Value;
    break;
  }
  case Delta12GOT: {
    assert(GOTSymbol && "No GOT section symbol");
    uint64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isUInt<12>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write16be(FixupPtr, (read16be(FixupPtr) & 0xF000) | Value);
    break;
  }
  case Delta64JumpSlot: {
    assert(GOTSymbol && "No GOT section symbol");
    uint64_t Value = S - GOTBase + A;
    *(big64_t *)FixupPtr = Value;
    break;
  }
  case Delta32JumpSlot: {
    assert(GOTSymbol && "No GOT section symbol");
    uint64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isUInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big32_t *)FixupPtr = Value;
    break;
  }
  case Delta20JumpSlot: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isInt<20>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write32be(FixupPtr, (read32be(FixupPtr) & 0xF00000FF) |
                            ((Value & 0xFFF) << 16) | ((Value & 0xFF000) >> 4));
    break;
  }
  case Delta16JumpSlot: {
    assert(GOTSymbol && "No GOT section symbol");
    uint64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isUInt<16>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    *(big16_t *)FixupPtr = Value;
    break;
  }
  case Delta12JumpSlot: {
    assert(GOTSymbol && "No GOT section symbol");
    uint64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isUInt<13>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write16be(FixupPtr, (read16be(FixupPtr) & 0xF000) | Value);
    break;
  }
  case PCRel32JumpSlot: {
    int64_t Value = S + A - P;
    if (!LLVM_UNLIKELY(isInt<33>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write32be(FixupPtr, Value >> 1);
    break;
  }
  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        " unsupported edge kind " + getEdgeKindName(E.getKind()));
  }

  return Error::success();
}

/// SystemZ null pointer content.
extern const char NullPointerContent[8];
inline ArrayRef<char> getGOTEntryBlockContent(LinkGraph &G) {
  return {reinterpret_cast<const char *>(NullPointerContent),
          G.getPointerSize()};
}

/// SystemZ pointer jump stub content.
///
/// Contains the instruction sequence for an indirect jump via an in-memory
/// pointer:
///   larl %r1, ptr
///   lg   %r1, 0(%r1)
///   j    %r1
constexpr size_t StubEntrySize = 14;
extern const char Pointer64JumpStubContent[StubEntrySize];
inline ArrayRef<char> getStubBlockContent(LinkGraph &G) {
  auto StubContent = Pointer64JumpStubContent;
  return {reinterpret_cast<const char *>(StubContent), StubEntrySize};
}

/// Creates a new pointer block in the given section and returns an
/// Anonymous symbol pointing to it.
///
/// If InitialTarget is given then an Pointer64 relocation will be added to the
/// block pointing at InitialTarget.
inline Symbol &createAnonymousPointer(LinkGraph &G, Section &PointerSection,
                                      Symbol *InitialTarget = nullptr,
                                      uint64_t InitialAddend = 0) {
  auto &B = G.createContentBlock(PointerSection, getGOTEntryBlockContent(G),
                                 orc::ExecutorAddr(), G.getPointerSize(), 0);
  if (InitialTarget)
    B.addEdge(Pointer64, 0, *InitialTarget, InitialAddend);
  return G.addAnonymousSymbol(B, 0, G.getPointerSize(), false, false);
}

/// Create a jump stub block that jumps via the pointer at the given symbol.
///
/// The stub block will have the following default values:
///   alignment: 8-bit
///   alignment-offset: 0
inline Block &createPointerJumpStubBlock(LinkGraph &G, Section &StubSection,
                                         Symbol &PointerSymbol) {
  auto &B = G.createContentBlock(StubSection, getStubBlockContent(G),
                                 orc::ExecutorAddr(), 8, 0);
  B.addEdge(Branch32dblToStub, 0, PointerSymbol, 0);
  return B;
}

/// Create a jump stub that jumps via the pointer at the given symbol and
/// an anonymous symbol pointing to it. Return the anonymous symbol.
///
/// The stub block will be created by createPointerJumpStubBlock.
inline Symbol &createAnonymousPointerJumpStub(LinkGraph &G,
                                              Section &StubSection,
                                              Symbol &PointerSymbol) {
  return G.addAnonymousSymbol(
      createPointerJumpStubBlock(G, StubSection, PointerSymbol), 0,
      StubEntrySize, true, false);
}

/// Global Offset Table Builder.
class GOTTableManager : public TableManager<GOTTableManager> {
public:
  static StringRef getSectionName() { return "$__GOT"; }

  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    if (E.getTarget().isDefined())
      return false;
    Edge::Kind KindToSet = Edge::Invalid;
    switch (E.getKind()) {
    case systemz::Delta12GOT:
    case systemz::Delta16GOT:
    case systemz::Delta20GOT:
    case systemz::Delta32GOT:
    case systemz::Delta64GOT:
    case systemz::Delta16FromGOT:
    case systemz::Delta32FromGOT:
    case systemz::Delta64FromGOT:
    case systemz::Delta12JumpSlot:
    case systemz::Delta16JumpSlot:
    case systemz::Delta32JumpSlot:
    case systemz::Delta64JumpSlot:
    case systemz::Delta20JumpSlot: {
    case systemz::DeltaPCRelGOT:
    case systemz::DeltaPCRelGOTdbl:
    case systemz::PCRel32GOTEntry:
    case systemz::PCRel32JumpSlot:
      KindToSet = E.getKind();
      break;
    }
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
    if (E.getTarget().isDefined())
      return false;

    switch (E.getKind()) {
    case systemz::BranchPCRelPLT32:
    case systemz::BranchPCRelPLT64:
    case systemz::BranchPCRelPLT12dbl:
    case systemz::BranchPCRelPLT16dbl:
    case systemz::BranchPCRelPLT24dbl:
    case systemz::BranchPCRelPLT32dbl:
    case systemz::DeltaPLT16FromGOT:
    case systemz::DeltaPLT32FromGOT:
    case systemz::DeltaPLT64FromGOT:
      break;
    default:
      return false;
    }
    DEBUG_WITH_TYPE("jitlink", {
      dbgs() << "  Fixing " << G.getEdgeKindName(E.getKind()) << " edge at "
             << B->getFixupAddress(E) << " (" << B->getAddress() << " + "
             << formatv("{0:x}", E.getOffset()) << ")\n";
    });
    E.setTarget(getEntryForTarget(G, E.getTarget()));
    return true;
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

} // namespace systemz
} // namespace jitlink
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_SYSTEMZ_H
