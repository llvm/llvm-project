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
  ///   - The target must reside in the low 20-bits of the address space,
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

  /// A 32-bit delta shifted by 1.
  ///
  /// Delta from the fixup to the target.
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
  Delta32dbl,

  /// A 24-bit delta shifted by 1.
  ///
  /// Delta from the fixup to the target.
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
  Delta24dbl,

  /// A 16-bit delta shifted by 1.
  ///
  /// Delta from the fixup to the target.
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
  Delta16dbl,

  /// A 12-bit delta shifted by 1.
  ///
  /// Delta from the fixup to the target.
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

  /// A 32-bit Delta shifted by 1.
  ///
  /// Delta from the fixup to the PLT slot for the target. This will lead to
  /// creation of a PLT stub.
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
  DeltaPLT32dbl,

  /// A 24-bit Delta shifted by 1.
  ///
  /// Delta from the fixup to the PLT slot for the target. This will lead to
  /// creation of a PLT stub.
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
  DeltaPLT24dbl,

  /// A 16-bit Delta shifted by 1.
  ///
  /// Delta from the fixup to the PLT slot for the target. This will lead to
  /// creation of a PLT stub.
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
  DeltaPLT16dbl,

  /// A 12-bit Delta shifted by 1.
  ///
  /// Delta from the fixup to the PLT slot for the target. This will lead to
  /// creation of a PLT stub.
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
  DeltaPLT12dbl,

  /// A 64-bit Delta.
  ///
  /// Delta from the fixup to the PLT slot for the target. This will lead to
  /// creation of a PLT stub.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend : int64
  ///
  DeltaPLT64,

  /// A 32-bit Delta.
  ///
  /// Delta from the fixup to the PLT slot for the target. This will lead to
  /// creation of a PLT stub.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  DeltaPLT32,

  /// A 64-bit offset from GOT to PLT.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int64
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///
  Delta64PLTFromGOT,

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
  Delta32PLTFromGOT,

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
  Delta16PLTFromGOT,

  /// A 64-bit offset from GOT.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int64
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///
  Delta64FromGOT,

  /// A 32-bit offset from GOT.
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
  Delta32FromGOT,

  /// A 16-bit offset from GOT.
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
  Delta16FromGOT,

  /// A 20-bit offset from GOT.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int20
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int16, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta20FromGOT,

  /// A 12-bit offset from GOT.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTBase + Addend : int12
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  ///   - The result of the fixup expression must fit into an int16, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta12FromGOT,

  /// A GOT entry getter/constructor, transformed to Delta64FromGOT pointing
  /// at the GOT entry for the original target.
  ///
  /// Indicates that this edge should be transformed into a Delta64FromGOT
  /// targeting the GOT entry for the edge's current target, maintaining the
  /// same addend. A GOT entry for the target should be created if one does
  /// not already exist.
  ///
  /// Edges of this kind are usually handled by a GOT builder pass inserted by
  /// default.
  ///
  /// Fixup expression:
  ///   NONE
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to handle edges of this kind prior to the fixup
  ///
  RequestGOTAndTransformToDelta64FromGOT,

  /// A GOT entry getter/constructor, transformed to Delta32FromGOT pointing
  /// at the GOT entry for the original target.
  ///
  /// Indicates that this edge should be transformed into a Delta32FromGOT
  /// targeting the GOT entry for the edge's current target, maintaining the
  /// same addend. A GOT entry for the target should be created if one does
  /// not already exist.
  ///
  /// Edges of this kind are usually handled by a GOT builder pass inserted by
  /// default.
  ///
  /// Fixup expression:
  ///   NONE
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to handle edges of this kind prior to the fixup
  ///
  RequestGOTAndTransformToDelta32FromGOT,

  /// A GOT entry getter/constructor, transformed to Delta20FromGOT pointing
  /// at the GOT entry for the original target.
  ///
  /// Indicates that this edge should be transformed into a Delta20FromGOT
  /// targeting the GOT entry for the edge's current target, maintaining the
  /// same addend. A GOT entry for the target should be created if one does
  /// not already exist.
  ///
  /// Edges of this kind are usually handled by a GOT builder pass inserted by
  /// default.
  ///
  /// Fixup expression:
  ///   NONE
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to handle edges of this kind prior to the fixup
  ///
  RequestGOTAndTransformToDelta20FromGOT,

  /// A GOT entry getter/constructor, transformed to Delta16FromGOT pointing
  /// at the GOT entry for the original target.
  ///
  /// Indicates that this edge should be transformed into a Delta16FromGOT
  /// targeting the GOT entry for the edge's current target, maintaining the
  /// same addend. A GOT entry for the target should be created if one does
  /// not already exist.
  ///
  /// Edges of this kind are usually handled by a GOT builder pass inserted by
  /// default.
  ///
  /// Fixup expression:
  ///   NONE
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to handle edges of this kind prior to the fixup
  ///
  RequestGOTAndTransformToDelta16FromGOT,

  /// A GOT entry getter/constructor, transformed to Delta12FromGOT pointing
  /// at the GOT entry for the original target.
  ///
  /// Indicates that this edge should be transformed into a Delta12FromGOT
  /// targeting the GOT entry for the edge's current target, maintaining the
  /// same addend. A GOT entry for the target should be created if one does
  /// not already exist.
  ///
  /// Edges of this kind are usually handled by a GOT builder pass inserted by
  /// default.
  ///
  /// Fixup expression:
  ///   NONE
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to handle edges of this kind prior to the fixup
  ///     phase will result in an assert/unreachable during the fixup phase.
  ///
  RequestGOTAndTransformToDelta12FromGOT,

  /// A GOT entry getter/constructor, transformed to Delta32dbl pointing at
  /// the GOT entry for the original target.
  ///
  /// Indicates that this edge should be transformed into a Delta32dbl targeting
  /// the GOT entry for the edge's current target, maintaining the same addend.
  /// A GOT entry for the target should be created if one does not already
  /// exist.
  ///
  /// Edges of this kind are usually handled by a GOT builder pass inserted by
  /// default.
  ///
  /// Fixup expression:
  ///   NONE
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to handle edges of this kind prior to the fixup
  ///     phase will result in an assert/unreachable during the fixup phase.
  ///
  RequestGOTAndTransformToDelta32dbl,

  /// A 32-bit Delta to GOT base.
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
  Delta32GOTBase,

  /// A 32-bit Delta to GOT base shifted by 1.
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
  Delta32dblGOTBase,

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
    write32be(FixupPtr, Value);
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
    write16be(FixupPtr, Value);
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
  case Delta64:
  case DeltaPLT64: {
    int64_t Value = S + A - P;
    write64be(FixupPtr, Value);
    break;
  }
  case Delta32:
  case DeltaPLT32: {
    int64_t Value = S + A - P;
    if (!LLVM_UNLIKELY(isInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write32be(FixupPtr, Value);
    break;
  }
  case Delta16: {
    int64_t Value = S + A - P;
    if (!LLVM_UNLIKELY(isInt<16>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write16be(FixupPtr, Value);
    break;
  }
  case NegDelta32: {
    int64_t Value = P + A - S;
    if (!LLVM_UNLIKELY(isInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write32be(FixupPtr, Value);
    break;
  }
  case Delta32dbl:
  case DeltaPLT32dbl: {
    int64_t Value = S + A - P;
    if (!LLVM_UNLIKELY(isInt<33>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write32be(FixupPtr, Value >> 1);
    break;
  }
  case Delta24dbl:
  case DeltaPLT24dbl: {
    int64_t Value = S + A - P;
    if (!LLVM_UNLIKELY(isInt<25>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    FixupPtr[0] = Value >> 17;
    FixupPtr[1] = Value >> 9;
    FixupPtr[2] = Value >> 1;
    break;
  }
  case Delta16dbl:
  case DeltaPLT16dbl: {
    int64_t Value = S + A - P;
    if (!LLVM_UNLIKELY(isInt<17>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write16be(FixupPtr, Value >> 1);
    break;
  }
  case Delta12dbl:
  case DeltaPLT12dbl: {
    int64_t Value = S + A - P;
    if (!LLVM_UNLIKELY(isInt<13>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write16be(FixupPtr,
              (read16be(FixupPtr) & 0xF000) | ((Value >> 1) & 0x0FFF));
    break;
  }
  case Delta32GOTBase: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = GOTBase + A - P;
    if (!LLVM_UNLIKELY(isInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write32be(FixupPtr, Value);
    break;
  }
  case Delta32dblGOTBase: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = GOTBase + A - P;
    if (!LLVM_UNLIKELY(isInt<33>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    if (!LLVM_UNLIKELY(isAlignmentCorrect(Value, 2)))
      return makeAlignmentError(FixupAddress, Value, 2, E);
    write32be(FixupPtr, Value >> 1);
    break;
  }
  case Delta64PLTFromGOT:
  case Delta64FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = S + A - GOTBase;
    write64be(FixupPtr, Value);
    break;
  }
  case Delta32PLTFromGOT:
  case Delta32FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = S + A - GOTBase;
    if (!LLVM_UNLIKELY(isInt<32>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write32be(FixupPtr, Value);
    break;
  }
  case Delta16PLTFromGOT:
  case Delta16FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    int64_t Value = S + A - GOTBase;
    if (!LLVM_UNLIKELY(isInt<16>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write16be(FixupPtr, Value);
    break;
  }
  case Delta20FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    uint64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isInt<20>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write32be(FixupPtr, (read32be(FixupPtr) & 0xF00000FF) |
                            ((Value & 0xFFF) << 16) | ((Value & 0xFF000) >> 4));
    break;
  }
  case Delta12FromGOT: {
    assert(GOTSymbol && "No GOT section symbol");
    uint64_t Value = S - GOTBase + A;
    if (!LLVM_UNLIKELY(isUInt<12>(Value)))
      return makeTargetOutOfRangeError(G, B, E);
    write16be(FixupPtr, (read16be(FixupPtr) & 0xF000) | Value);
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
///   lgrl %r1, ptr
///   j    %r1
constexpr size_t StubEntrySize = 8;
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
///   alignment: 16-bit
///   alignment-offset: 0
inline Block &createPointerJumpStubBlock(LinkGraph &G, Section &StubSection,
                                         Symbol &PointerSymbol) {
  auto &B = G.createContentBlock(StubSection, getStubBlockContent(G),
                                 orc::ExecutorAddr(), 16, 0);
  B.addEdge(Delta32dbl, 2, PointerSymbol, 2);
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
    case systemz::RequestGOTAndTransformToDelta12FromGOT:
      KindToSet = systemz::Delta12FromGOT;
      break;
    case systemz::RequestGOTAndTransformToDelta16FromGOT:
      KindToSet = systemz::Delta16FromGOT;
      break;
    case systemz::RequestGOTAndTransformToDelta20FromGOT:
      KindToSet = systemz::Delta20FromGOT;
      break;
    case systemz::RequestGOTAndTransformToDelta32FromGOT:
      KindToSet = systemz::Delta32FromGOT;
      break;
    case systemz::RequestGOTAndTransformToDelta64FromGOT:
      KindToSet = systemz::Delta64FromGOT;
      break;
    case systemz::RequestGOTAndTransformToDelta32dbl:
      KindToSet = systemz::DeltaPLT32dbl;
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
    if (E.getTarget().isDefined())
      return false;

    switch (E.getKind()) {
    case systemz::DeltaPLT32:
    case systemz::DeltaPLT64:
    case systemz::DeltaPLT12dbl:
    case systemz::DeltaPLT16dbl:
    case systemz::DeltaPLT24dbl:
    case systemz::DeltaPLT32dbl:
    case systemz::Delta16PLTFromGOT:
    case systemz::Delta32PLTFromGOT:
    case systemz::Delta64PLTFromGOT:
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
