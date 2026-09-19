//===- EHPersonalities.h - Compute EH-related information -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_EHPERSONALITIES_H
#define LLVM_IR_EHPERSONALITIES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {
class BasicBlock;
class Function;
class Triple;
class Value;

enum class EHPersonality {
  Unknown,
  GNU_Ada,
  GNU_C,
  GNU_C_SjLj,
  GNU_CXX,
  GNU_CXX_SjLj,
  GNU_ObjC,
  MSVC_X86SEH,
  MSVC_TableSEH,
  MSVC_CXX,
  CoreCLR,
  Rust,
  Wasm_CXX,
  XL_CXX,
  ZOS_CXX,
};

/// See if the given exception handling personality function is one
/// that we understand.  If so, return a description of it; otherwise return
/// Unknown.
LLVM_ABI EHPersonality classifyEHPersonality(const Value *Pers);

LLVM_ABI StringRef getEHPersonalityName(EHPersonality Pers);

LLVM_ABI EHPersonality getDefaultEHPersonality(const Triple &T);

/// Returns true if this personality function catches asynchronous
/// exceptions.
inline bool isAsynchronousEHPersonality(EHPersonality Pers) {
  // The two SEH personality functions can catch asynch exceptions. We assume
  // unknown personalities don't catch asynch exceptions.
  switch (Pers) {
  case EHPersonality::MSVC_X86SEH:
  case EHPersonality::MSVC_TableSEH:
    return true;
  default:
    return false;
  }
  llvm_unreachable("invalid enum");
}

/// Returns true if this is a personality function that invokes
/// handler funclets (which must return to it).
inline bool isFuncletEHPersonality(EHPersonality Pers) {
  switch (Pers) {
  case EHPersonality::MSVC_CXX:
  case EHPersonality::MSVC_X86SEH:
  case EHPersonality::MSVC_TableSEH:
  case EHPersonality::CoreCLR:
    return true;
  default:
    return false;
  }
  llvm_unreachable("invalid enum");
}

/// Returns true if this personality uses scope-style EH IR instructions:
/// catchswitch, catchpad/ret, and cleanuppad/ret.
inline bool isScopedEHPersonality(EHPersonality Pers) {
  switch (Pers) {
  case EHPersonality::MSVC_CXX:
  case EHPersonality::MSVC_X86SEH:
  case EHPersonality::MSVC_TableSEH:
  case EHPersonality::CoreCLR:
  case EHPersonality::Wasm_CXX:
    return true;
  default:
    return false;
  }
  llvm_unreachable("invalid enum");
}

/// Return true if this personality may be safely removed if there
/// are no invoke instructions remaining in the current function.
inline bool isNoOpWithoutInvoke(EHPersonality Pers) {
  switch (Pers) {
  case EHPersonality::Unknown:
    return false;
  // All known personalities currently have this behavior
  default:
    return true;
  }
  llvm_unreachable("invalid enum");
}

LLVM_ABI bool canSimplifyInvokeNoUnwind(const Function *F);

/// Snapshot of the protecting EH pad for blocks in table-based asynchronous
/// SEH. Scope-table dispatch depends on where an instruction executes, so
/// moving it between regions can change the handler that observes a fault.
/// Recompute this information after changing the CFG or the try markers.
class LLVM_ABI SEHTryRegionInfo {
public:
  explicit SEHTryRegionInfo(const Function &F);

  /// Whether region membership permits the move, not a complete legality check.
  /// With active region tracking, different blocks must have known, equal
  /// regions. Without applicable markers, this query imposes no restriction.
  bool isSameRegion(const BasicBlock *From, const BasicBlock *To) const;

private:
  // An empty map disables this analysis. Otherwise, an absent key is unvisited,
  // nullopt is ambiguous, and an engaged null pointer means unwind to caller.
  DenseMap<const BasicBlock *, std::optional<const BasicBlock *>> RegionOf;
};

typedef TinyPtrVector<BasicBlock *> ColorVector;

/// If an EH funclet personality is in use (see isFuncletEHPersonality),
/// this will recompute which blocks are in which funclet. It is possible that
/// some blocks are in multiple funclets. Consider this analysis to be
/// expensive.
LLVM_ABI DenseMap<BasicBlock *, ColorVector> colorEHFunclets(Function &F);

} // end namespace llvm

#endif // LLVM_IR_EHPERSONALITIES_H
