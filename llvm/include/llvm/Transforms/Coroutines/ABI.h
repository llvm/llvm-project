//===- ABI.h - Coroutine lowering class definitions (ABIs) ----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines coroutine lowering classes. The interface for coroutine
// lowering is defined by BaseABI. Each lowering method (ABI) implements the
// interface. Note that the enum class ABI, such as ABI::Switch, determines
// which ABI class, such as SwitchABI, is used to lower the coroutine. Both the
// ABI enum and ABI class are used by the Coroutine passes when lowering.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_COROUTINES_ABI_H
#define LLVM_TRANSFORMS_COROUTINES_ABI_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Coroutines/CoroShape.h"
#include "llvm/Transforms/Coroutines/MaterializationUtils.h"
#include "llvm/Transforms/Coroutines/SpillUtils.h"

namespace llvm {

class Function;

namespace coro {

// Mapping from the to-be-spilled value to all the users that need reload.
struct FrameDataInfo {
  // All the values (that are not allocas) that needs to be spilled to the
  // frame.
  coro::SpillInfo Spills;
  // Allocas contains all values defined as allocas that need to live in the
  // frame.
  SmallVector<coro::AllocaInfo, 8> Allocas;
  // Map SSA values to corresponding GEPs (possibly with casts) to frame
  SmallMapVector<Value *, SmallVector<Instruction *, 2>, 8> SpillGepMap;

  FrameDataInfo() = default;
  FrameDataInfo(coro::SpillInfo Spills,
                SmallVector<coro::AllocaInfo, 8> Allocas)
      : Spills(std::move(Spills)), Allocas(std::move(Allocas)) {}

  SmallVector<Value *, 8> getAllDefs() const {
    SmallVector<Value *, 8> Defs;
    for (const auto &P : Spills)
      Defs.push_back(P.first);
    for (const auto &A : Allocas)
      Defs.push_back(A.Alloca);
    return Defs;
  }

  uint32_t getFieldIndex(Value *V) const {
    auto Itr = FieldIndexMap.find(V);
    assert(Itr != FieldIndexMap.end() &&
           "Value does not have a frame field index");
    return Itr->second;
  }

  void setFieldIndex(Value *V, uint32_t Index) {
    assert(FieldIndexMap.count(V) == 0 &&
           "Cannot set the index for the same field twice.");
    FieldIndexMap[V] = Index;
  }

  Align getAlign(Value *V) const {
    auto Iter = FieldAlignMap.find(V);
    assert(Iter != FieldAlignMap.end());
    return Iter->second;
  }

  void setAlign(Value *V, Align AL) {
    assert(FieldAlignMap.count(V) == 0);
    FieldAlignMap.insert({V, AL});
  }

  uint64_t getDynamicAlign(Value *V) const {
    auto Iter = FieldDynamicAlignMap.find(V);
    assert(Iter != FieldDynamicAlignMap.end());
    return Iter->second;
  }

  void setDynamicAlign(Value *V, uint64_t Align) {
    assert(FieldDynamicAlignMap.count(V) == 0);
    FieldDynamicAlignMap.insert({V, Align});
  }

  uint64_t getOffset(Value *V) const {
    auto Iter = FieldOffsetMap.find(V);
    assert(Iter != FieldOffsetMap.end());
    return Iter->second;
  }

  void setOffset(Value *V, uint64_t Offset) {
    assert(FieldOffsetMap.count(V) == 0);
    FieldOffsetMap.insert({V, Offset});
  }

private:
  // Map from values to their slot indexes on the frame (insertion order).
  DenseMap<Value *, uint32_t> FieldIndexMap;
  // Map from values to their alignment on the frame. They would be set after
  // the frame is built.
  DenseMap<Value *, Align> FieldAlignMap;
  DenseMap<Value *, uint64_t> FieldDynamicAlignMap;
  // Map from values to their offset on the frame. They would be set after
  // the frame is built.
  DenseMap<Value *, uint64_t> FieldOffsetMap;
};

// This interface/API is to provide an object oriented way to implement ABI
// functionality. This is intended to replace use of the ABI enum to perform
// ABI operations. The ABIs (e.g. Switch, Async, Retcon{Once}) are the common
// ABIs. However, specific users may need to modify the behavior of these. This
// can be accomplished by inheriting one of the common ABIs and overriding one
// or more of the methods to create a custom ABI. To use a custom ABI for a
// given coroutine the coro.begin.custom.abi intrinsic is used in place of the
// coro.begin intrinsic. This takes an additional i32 arg that specifies the
// index of an ABI generator for the custom ABI object in a SmallVector passed
// to CoroSplitPass ctor.

class LLVM_ABI BaseABI {
public:
  BaseABI(Function &F, coro::Shape &S,
          std::function<bool(Instruction &)> IsMaterializable)
      : F(F), Shape(S), IsMaterializable(std::move(IsMaterializable)) {}
  virtual ~BaseABI() = default;

  // Initialize the coroutine ABI
  virtual void init() = 0;

  // Allocate the coroutine frame and do spill/reload as needed.
  virtual void buildCoroutineFrame(bool OptimizeFrame);

  // Perform the function splitting according to the ABI.
  virtual void splitCoroutine(Function &F, coro::Shape &Shape,
                              SmallVectorImpl<Function *> &Clones,
                              TargetTransformInfo &TTI) = 0;

  Function &F;
  coro::Shape &Shape;
  FrameDataInfo FrameData;

  // Callback used by coro::BaseABI::buildCoroutineFrame for rematerialization.
  // It is provided to coro::doMaterializations(..).
  std::function<bool(Instruction &I)> IsMaterializable;

protected:
  void remapReloadToSSA();
};

class LLVM_ABI SwitchABI : public BaseABI {
public:
  SwitchABI(Function &F, coro::Shape &S,
            std::function<bool(Instruction &)> IsMaterializable)
      : BaseABI(F, S, std::move(IsMaterializable)) {}

  void init() override;

  void splitCoroutine(Function &F, coro::Shape &Shape,
                      SmallVectorImpl<Function *> &Clones,
                      TargetTransformInfo &TTI) override;
};

class LLVM_ABI AsyncABI : public BaseABI {
public:
  AsyncABI(Function &F, coro::Shape &S,
           std::function<bool(Instruction &)> IsMaterializable)
      : BaseABI(F, S, std::move(IsMaterializable)) {}

  void init() override;

  void splitCoroutine(Function &F, coro::Shape &Shape,
                      SmallVectorImpl<Function *> &Clones,
                      TargetTransformInfo &TTI) override;
};

class LLVM_ABI AnyRetconABI : public BaseABI {
public:
  AnyRetconABI(Function &F, coro::Shape &S,
               std::function<bool(Instruction &)> IsMaterializable)
      : BaseABI(F, S, std::move(IsMaterializable)) {}

  void init() override;

  void splitCoroutine(Function &F, coro::Shape &Shape,
                      SmallVectorImpl<Function *> &Clones,
                      TargetTransformInfo &TTI) override;
};

} // end namespace coro

} // end namespace llvm

#endif // LLVM_TRANSFORMS_COROUTINES_ABI_H
