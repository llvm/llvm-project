//===- InstrMaps.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_INSTRMAPS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_INSTRMAPS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm::sandboxir {

class LegalityResult;

struct Action {
  unsigned Idx = 0;
  const LegalityResult *LegalityRes = nullptr;
  SmallVector<Value *, 4> Bndl;
  SmallVector<Value *> UserBndl;
  unsigned Depth;
  SmallVector<Action *> Operands;
  Value *Vec = nullptr;
  Action(const LegalityResult *LR, ArrayRef<Value *> B, ArrayRef<Value *> UB,
         unsigned Depth)
      : LegalityRes(LR), Bndl(B), UserBndl(UB), Depth(Depth) {}
#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const Action &A) {
    A.print(OS);
    return OS;
  }
#endif // NDEBUG
};

/// Maps the original instructions to the vectorized instrs and the reverse.
/// For now an original instr can only map to a single vector.
class InstrMaps {
  /// A map from the original values that got combined into vectors, to the
  /// vectorization Action.
  DenseMap<Value *, Action *> OrigToVectorMap;
  /// A map from the vec Action to a map of the original value to its lane.
  /// Please note that for constant vectors, there may multiple original values
  /// with the same lane, as they may be coming from vectorizing different
  /// original values.
  DenseMap<Action *, DenseMap<Value *, unsigned>> VectorToOrigLaneMap;
  std::optional<Context::CallbackID> EraseInstrCB;

public:
  InstrMaps() = default;
  ~InstrMaps() = default;
  /// \Returns the vector value that we got from vectorizing \p Orig, or
  /// nullptr if not found.
  Action *getVectorForOrig(Value *Orig) const {
    auto It = OrigToVectorMap.find(Orig);
    return It != OrigToVectorMap.end() ? It->second : nullptr;
  }
  /// \Returns the lane of \p Orig before it got vectorized into \p Vec, or
  /// nullopt if not found.
  std::optional<unsigned> getOrigLane(Action *Vec, Value *Orig) const {
    auto It1 = VectorToOrigLaneMap.find(Vec);
    if (It1 == VectorToOrigLaneMap.end())
      return std::nullopt;
    const auto &OrigToLaneMap = It1->second;
    auto It2 = OrigToLaneMap.find(Orig);
    if (It2 == OrigToLaneMap.end())
      return std::nullopt;
    return It2->second;
  }
  /// Update the map to reflect that \p Origs got vectorized into \p Vec.
  void registerVector(ArrayRef<Value *> Origs, Action *Vec) {
    auto &OrigToLaneMap = VectorToOrigLaneMap[Vec];
    unsigned Lane = 0;
    for (Value *Orig : Origs) {
      auto Pair = OrigToVectorMap.try_emplace(Orig, Vec);
      assert(Pair.second && "Orig already exists in the map!");
      (void)Pair;
      OrigToLaneMap[Orig] = Lane;
      Lane += VecUtils::getNumLanes(Orig);
    }
  }
  void clear() {
    OrigToVectorMap.clear();
    VectorToOrigLaneMap.clear();
  }
#ifndef NDEBUG
  void print(raw_ostream &OS) const {
    OS << "OrigToVectorMap:\n";
    for (auto [Orig, Vec] : OrigToVectorMap)
      OS << *Orig << " : " << *Vec << "\n";
  }
  LLVM_DUMP_METHOD void dump() const;
#endif
};
} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_INSTRMAPS_H
