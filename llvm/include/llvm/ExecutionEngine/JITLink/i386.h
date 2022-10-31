//=== i386.h - Generic JITLink i386 edge kinds, utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing i386 objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_I386_H
#define LLVM_EXECUTIONENGINE_JITLINK_I386_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/TableManager.h"

namespace llvm::jitlink::i386 {
/// Represets i386 fixups
enum EdgeKind_i386 : Edge::Kind {

  /// None
  None = Edge::FirstRelocation,

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

  /// A 32-bit PC-relative relocation.
  ///
  /// Represents a data/control flow instruction using PC-relative addressing
  /// to a target.
  ///
  /// The fixup expression for this kind includes an implicit offset to account
  /// for the PC (unlike the Delta edges) so that a PCRel32 with a target
  /// T and addend zero is a call/branch to the start (offset zero) of T.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - (Fixup + 4) + Addend : int32
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int32, otherwise
  ///     an out-of-range error will be returned.
  ///
  PCRel32,

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

  /// A 16-bit PC-relative relocation.
  ///
  /// Represents a data/control flow instruction using PC-relative addressing
  /// to a target.
  ///
  /// The fixup expression for this kind includes an implicit offset to account
  /// for the PC (unlike the Delta edges) so that a PCRel16 with a target
  /// T and addend zero is a call/branch to the start (offset zero) of T.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - (Fixup + 4) + Addend : int16
  ///
  /// Errors:
  ///   - The result of the fixup expression must fit into an int16, otherwise
  ///     an out-of-range error will be returned.
  ///
  PCRel16,

  /// A 64-bit GOT delta.
  ///
  /// Delta from the global offset table to the target.
  ///
  /// Fixup expression:
  ///   Fixup <- Target - GOTSymbol + Addend : int32
  ///
  /// Errors:
  ///   - *ASSERTION* Failure to a null pointer GOTSymbol, which the GOT section
  ///     symbol was not been defined.
  Delta32FromGOT,
};

/// Returns a string name for the given i386 edge. For debugging purposes
/// only
const char *getEdgeKindName(Edge::Kind K);

/// Returns true if the given uint32_t value is in range for a uint16_t.
inline bool isInRangeForImmU16(uint32_t Value) {
  return Value <= std::numeric_limits<uint16_t>::max();
}

/// Returns true if the given int32_t value is in range for an int16_t.
inline bool isInRangeForImmS16(int32_t Value) {
  return (Value >= std::numeric_limits<int16_t>::min() &&
          Value <= std::numeric_limits<int16_t>::max());
}

/// Apply fixup expression for edge to block content.
inline Error applyFixup(LinkGraph &G, Block &B, const Edge &E) {
  using namespace i386;
  using namespace llvm::support;

  char *BlockWorkingMem = B.getAlreadyMutableContent().data();
  char *FixupPtr = BlockWorkingMem + E.getOffset();
  auto FixupAddress = B.getAddress() + E.getOffset();

  switch (E.getKind()) {
  case i386::None: {
    break;
  }

  case i386::Pointer32: {
    uint32_t Value = E.getTarget().getAddress().getValue() + E.getAddend();
    *(ulittle32_t *)FixupPtr = Value;
    break;
  }

  case i386::PCRel32: {
    int32_t Value =
        E.getTarget().getAddress() - (FixupAddress + 4) + E.getAddend();
    *(little32_t *)FixupPtr = Value;
    break;
  }

  case i386::Pointer16: {
    uint32_t Value = E.getTarget().getAddress().getValue() + E.getAddend();
    if (LLVM_LIKELY(isInRangeForImmU16(Value)))
      *(ulittle16_t *)FixupPtr = Value;
    else
      return makeTargetOutOfRangeError(G, B, E);
    break;
  }

  case i386::PCRel16: {
    int32_t Value =
        E.getTarget().getAddress() - (FixupAddress + 4) + E.getAddend();
    if (LLVM_LIKELY(isInRangeForImmS16(Value)))
      *(little16_t *)FixupPtr = Value;
    else
      return makeTargetOutOfRangeError(G, B, E);
    break;
  }

  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        "unsupported edge kind" + getEdgeKindName(E.getKind()));
  }

  return Error::success();
}
} // namespace llvm::jitlink::i386

#endif // LLVM_EXECUTIONENGINE_JITLINK_I386_H
