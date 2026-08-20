//===- llvm/IR/IRUnitRef.h - Reference to an IR unit ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines IRUnitRef, a type-erased reference to the IR unit a pass
/// or analysis is running on, and IRUnitKindTraits, which IR units specialize
/// to opt into being referred to by one.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_IRUNITREF_H
#define LLVM_IR_IRUNITREF_H

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

/// A type-erased reference to the IR unit a pass or analysis is running on,
/// together with the kind of IR unit it refers to.
class IRUnitRef {
  template <typename To, typename From, typename Enable> friend struct CastInfo;

  const void *Ptr;
  IRUnitKind Kind;

  /// Which kind of IR unit is wrapped.
  IRUnitKind getKind() const { return Kind; }

  /// The wrapped IR unit, type-erased.
  const void *getPointer() const { return Ptr; }

public:
  template <typename IRUnitT, IRUnitKind K = IRUnitKindTraits<IRUnitT>::Kind>
  IRUnitRef(const IRUnitT &IR) : Ptr(&IR), Kind(K) {}
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
