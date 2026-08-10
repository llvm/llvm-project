//===- llvm/IR/IRUnitRef.h - Reference to an IR unit ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines IRUnitRef, a tagged pointer to the IR unit a pass or
/// analysis is running on, and IRUnitKindTraits, which IR units specialize to
/// opt into being referred to by one.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_IRUNITREF_H
#define LLVM_IR_IRUNITREF_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Casting.h"
#include <cstddef>
#include <type_traits>

namespace llvm {

class Function;
class Loop;
class MachineFunction;
class Module;

/// The IR units a pass can run on.
enum class IRUnitKind {
  Module,
  Function,
  Loop,
  MachineFunction,
  LazyCallGraphSCC,
};

/// Map an IR unit type to its IRUnitKind.
template <typename IRUnitT> struct IRUnitKindTraits {};

template <> struct IRUnitKindTraits<Module> {
  static constexpr IRUnitKind Kind = IRUnitKind::Module;
};
template <> struct IRUnitKindTraits<Function> {
  static constexpr IRUnitKind Kind = IRUnitKind::Function;
};
template <> struct IRUnitKindTraits<Loop> {
  static constexpr IRUnitKind Kind = IRUnitKind::Loop;
};
template <> struct IRUnitKindTraits<MachineFunction> {
  static constexpr IRUnitKind Kind = IRUnitKind::MachineFunction;
};

// IRUnitKindTraits<LazyCallGraph::SCC> needs to be defined in
// Analysis/LazyCallGraph.h.

/// The number of bits IRUnitKind needs.
static constexpr int IRUnitKindBits = 3;

/// Pointer traits for the type-erased IR unit held by IRUnitRef.
///
/// IR units are not complete at this point, so only assume minimum pointer
/// alignment.
struct IRUnitPtrTraits {
  static void *getAsVoidPointer(const void *P) { return const_cast<void *>(P); }
  static const void *getFromVoidPointer(void *P) { return P; }

  static constexpr int NumLowBitsAvailable = ConstantLog2<alignof(void *)>();
};

/// A tagged pointer to the IR unit a pass or analysis is running on.
class IRUnitRef {
  /// Whether the kind fits in the spare low bits of an IR unit pointer.
  static constexpr bool KindInPointer =
      IRUnitPtrTraits::NumLowBitsAvailable >= IRUnitKindBits;

  /// Unpacked storage for IR unit and kind with interface matching
  /// PointerIntPair. so that the two are interchangeable here.
  class UnpackedPointerAndKind {
    const void *Ptr;
    IRUnitKind Kind;

  public:
    UnpackedPointerAndKind(const void *Ptr, IRUnitKind Kind)
        : Ptr(Ptr), Kind(Kind) {}

    const void *getPointer() const { return Ptr; }
    IRUnitKind getInt() const { return Kind; }
  };

  /// Use PointerIntPair when enough bits are available, fall back to
  /// UnpackedPointerAndKind otherwise.
  std::conditional_t<
      KindInPointer,
      PointerIntPair<const void *, IRUnitKindBits, IRUnitKind, IRUnitPtrTraits>,
      UnpackedPointerAndKind>
      Value;

public:
  template <typename IRUnitT, IRUnitKind K = IRUnitKindTraits<IRUnitT>::Kind>
  IRUnitRef(const IRUnitT &IR) : Value(&IR, K) {
    static_assert(!KindInPointer ||
                      ConstantLog2<alignof(IRUnitT)>() >= IRUnitKindBits,
                  "IR unit is not aligned enough to hold the kind");
  }

  /// Which kind of IR unit is wrapped. Prefer isa/cast/dyn_cast.
  IRUnitKind getKind() const { return Value.getInt(); }

  /// The wrapped IR unit, type-erased. Prefer isa/cast/dyn_cast.
  const void *getPointer() const { return Value.getPointer(); }
};

static_assert(!std::is_constructible_v<IRUnitRef, std::nullptr_t>,
              "IRUnitRef must not be constructible from nullptr");

/// Lets isa/cast/dyn_cast query which IR unit an IRUnitRef holds, naming the IR
/// unit itself rather than a pointer to it: dyn_cast<Module>(IR).
template <typename To> struct CastInfo<To, IRUnitRef> {
  static bool isPossible(IRUnitRef IR) {
    return IR.getKind() == IRUnitKindTraits<To>::Kind;
  }

  static const To *doCast(IRUnitRef IR) {
    return static_cast<const To *>(IR.getPointer());
  }

  static const To *castFailed() { return nullptr; }

  static const To *doCastIfPossible(IRUnitRef IR) {
    return isPossible(IR) ? doCast(IR) : castFailed();
  }
};

template <typename To>
struct CastInfo<To, const IRUnitRef> : public CastInfo<To, IRUnitRef> {};

} // end namespace llvm

#endif // LLVM_IR_IRUNITREF_H
