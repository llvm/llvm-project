//===-- x86_64.h - Generic JITLink x86-64 edge kinds, utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing x86-64 objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_X86_64_H
#define LLVM_EXECUTIONENGINE_JITLINK_X86_64_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

#include <limits>

namespace llvm {
namespace jitlink {
namespace x86_64 {

/// Represents x86-64 fixups and other x86-64-specific edge kinds.
enum EdgeKind_x86_64 : Edge::Kind {

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
  ///   Fixup <- Target - Fixup + Addend : int64
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  Delta32,

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

  /// A 32-bit PC-relative branch.
  ///
  /// Represents a PC-relative call or branch to a target. This can be used to
  /// identify, record, and/or patch call sites.
  ///
  /// The fixup expression for this kind includes an implicit offset to account
  /// for the PC (unlike the Delta edges) so that a Branch32PCRel with a target
  /// T and addend zero is a call/branch to the start (offset zero) of T.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - (Fixup + 4) + Addend : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  BranchPCRel32,

  /// A 32-bit PC-relative branch to a pointer jump stub.
  ///
  /// The target of this relocation should be a pointer jump stub of the form:
  ///
  /// \code{.s}
  ///   .text
  ///   jmpq *tgtptr(%rip)
  ///   ; ...
  ///
  ///   .data
  ///   tgtptr:
  ///     .quad 0
  /// \endcode
  ///
  /// This edge kind has the same fixup expression as BranchPCRel32, but further
  /// identifies the call/branch as being to a pointer jump stub. For edges of
  /// this kind the jump stub should not be bypassed (use
  /// BranchPCRel32ToPtrJumpStubRelaxable for that), but the pointer location
  /// target may be recorded to allow manipulation at runtime.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend - 4 : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  BranchPCRel32ToPtrJumpStub,

  /// A relaxable version of BranchPCRel32ToPtrJumpStub.
  ///
  /// The edge kind has the same fixup expression as BranchPCRel32ToPtrJumpStub,
  /// but identifies the call/branch as being to a pointer jump stub that may be
  /// bypassed if the ultimate target is within range of the fixup location.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend - 4: int32
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  BranchPCRel32ToPtrJumpStubRelaxable,

  /// A GOT entry getter/constructor, transformed to Delta32 pointing at the GOT
  /// entry for the original target.
  ///
  /// Indicates that this edge should be transformed into a Delta32 targeting
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
  RequestGOTAndTransformToDelta32,

  /// A PC-relative reference to a GOT entry, relaxable if GOT entry target
  /// is in-range of the fixup.
  ///
  /// If the GOT entry target is in-range of the fixup then the load from the
  /// GOT may be replaced with a direct memory address calculation.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - (Fixup + 4) + Addend : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  PCRel32GOTLoadRelaxable,

  /// A GOT entry getter/constructor, transformed to PCRel32ToGOTLoadRelaxable
  /// pointing at the GOT entry for the original target.
  ///
  /// Indicates that this edge should be transformed into a
  /// PC32ToGOTLoadRelaxable targeting the GOT entry for the edge's current
  /// target, maintaining the same addend. A GOT entry for the target should be
  /// created if one does not already exist.
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
  RequestGOTAndTransformToPCRel32GOTLoadRelaxable,

  /// A PC-relative reference to a Thread Local Variable Pointer (TLVP) entry,
  /// relaxable if the TLVP entry target is in-range of the fixup.
  ///
  /// If the TLVP entry target is in-range of the fixup then the load frmo the
  /// TLVP may be replaced with a direct memory address calculation.
  ///
  /// The target of this edge must be a thread local variable entry of the form
  ///   .quad <tlv getter thunk>
  ///   .quad <tlv key>
  ///   .quad <tlv initializer>
  ///
  /// Fixup expression:
  ///   Fixup <- Target - (Fixup + 4) + Addend : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///   - The target must be either external, or a TLV entry of the required
  ///     form, otherwise a malformed TLV entry error will be returned.
  ///
  PCRel32TLVPLoadRelaxable,

  /// A TLVP entry getter/constructor, transformed to
  /// Delta32ToTLVPLoadRelaxable.
  ///
  /// Indicates that this edge should be transformed into a
  /// Delta32ToTLVPLoadRelaxable targeting the TLVP entry for the edge's current
  /// target. A TLVP entry for the target should be created if one does not
  /// already exist.
  ///
  /// Fixup expression:
  ///   NONE
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to handle edges of this kind prior to the fixup
  ///     phase will result in an assert/unreachable during the fixup phase.
  ///
  RequestTLVPAndTransformToPCRel32TLVPLoadRelaxable
};

/// Returns a string name for the given x86-64 edge. For debugging purposes
/// only.
const char *getEdgeKindName(Edge::Kind K);

/// Returns true if the given uint64_t value is in range for a uint32_t.
inline bool isInRangeForImmU32(uint64_t Value) {
  return Value <= std::numeric_limits<uint32_t>::max();
}

/// Returns true if the given int64_t value is in range for an int32_t.
inline bool isInRangeForImmS32(int64_t Value) {
  return (Value >= std::numeric_limits<int32_t>::min() &&
          Value <= std::numeric_limits<int32_t>::max());
}

/// Apply fixup expression for edge to block content.
inline Error applyFixup(LinkGraph &G, Block &B, const Edge &E,
                        char *BlockWorkingMem) {
  using namespace support;

  char *FixupPtr = BlockWorkingMem + E.getOffset();
  JITTargetAddress FixupAddress = B.getAddress() + E.getOffset();

  switch (E.getKind()) {

  case Pointer64: {
    uint64_t Value = E.getTarget().getAddress() + E.getAddend();
    *(ulittle64_t *)FixupPtr = Value;
    break;
  }

  case Pointer32: {
    uint64_t Value = E.getTarget().getAddress() + E.getAddend();
    if (LLVM_LIKELY(isInRangeForImmU32(Value)))
      *(ulittle32_t *)FixupPtr = Value;
    else
      return makeTargetOutOfRangeError(G, B, E);
    break;
  }

  case BranchPCRel32:
  case BranchPCRel32ToPtrJumpStub:
  case BranchPCRel32ToPtrJumpStubRelaxable:
  case PCRel32GOTLoadRelaxable:
  case PCRel32TLVPLoadRelaxable: {
    int64_t Value =
        E.getTarget().getAddress() - (FixupAddress + 4) + E.getAddend();
    if (LLVM_LIKELY(isInRangeForImmS32(Value)))
      *(little32_t *)FixupPtr = Value;
    else
      return makeTargetOutOfRangeError(G, B, E);
    break;
  }

  case Delta64: {
    int64_t Value = E.getTarget().getAddress() - FixupAddress + E.getAddend();
    *(little64_t *)FixupPtr = Value;
    break;
  }

  case Delta32: {
    int64_t Value = E.getTarget().getAddress() - FixupAddress + E.getAddend();
    if (LLVM_LIKELY(isInRangeForImmS32(Value)))
      *(little32_t *)FixupPtr = Value;
    else
      return makeTargetOutOfRangeError(G, B, E);
    break;
  }

  case NegDelta64: {
    int64_t Value = FixupAddress - E.getTarget().getAddress() + E.getAddend();
    *(little64_t *)FixupPtr = Value;
    break;
  }

  case NegDelta32: {
    int64_t Value = FixupAddress - E.getTarget().getAddress() + E.getAddend();
    if (LLVM_LIKELY(isInRangeForImmS32(Value)))
      *(little32_t *)FixupPtr = Value;
    else
      return makeTargetOutOfRangeError(G, B, E);
    break;
  }

  default: {
    // If you hit this you should check that *constructor and other non-fixup
    // edges have been removed prior to applying fixups.
    llvm_unreachable("Graph contains edge kind with no fixup expression");
  }
  }

  return Error::success();
}

/// x86-64 null pointer content.
extern const char NullPointerContent[8];

/// x86-64 pointer jump stub content.
///
/// Contains the instruction sequence for an indirect jump via an in-memory
/// pointer:
///   jmpq *ptr(%rip)
extern const char PointerJumpStubContent[6];

/// Creates a new pointer block in the given section and returns an anonymous
/// symbol pointing to it.
///
/// If InitialTarget is given then an Pointer64 relocation will be added to the
/// block pointing at InitialTarget.
///
/// The pointer block will have the following default values:
///   alignment: 64-bit
///   alignment-offset: 0
///   address: highest allowable (~7U)
inline Symbol &createAnonymousPointer(LinkGraph &G, Section &PointerSection,
                                      Symbol *InitialTarget = nullptr,
                                      uint64_t InitialAddend = 0) {
  auto &B =
      G.createContentBlock(PointerSection, NullPointerContent, ~7ULL, 8, 0);
  if (InitialTarget)
    B.addEdge(Pointer64, 0, *InitialTarget, InitialAddend);
  return G.addAnonymousSymbol(B, 0, 8, false, false);
}

/// Create a jump stub that jumps via the pointer at the given symbol, returns
/// an anonymous symbol pointing to it.
///
/// The stub block will have the following default values:
///   alignment: 8-bit
///   alignment-offset: 0
///   address: highest allowable: (~5U)
inline Symbol &createAnonymousPointerJumpStub(LinkGraph &G,
                                              Section &StubSection,
                                              Symbol &PointerSymbol) {
  auto &B =
      G.createContentBlock(StubSection, PointerJumpStubContent, ~5ULL, 1, 0);
  B.addEdge(Delta32, 2, PointerSymbol, -4);
  return G.addAnonymousSymbol(B, 0, 6, true, false);
}

} // namespace x86_64
} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_X86_64_H
