//=== WebAssemblyInstCombineIntrinsic.cpp -
//                                WebAssembly specific InstCombine pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to
/// WebAssembly. It uses the target's detailed information to provide more
/// precise answers to certain TTI queries, while letting the target independent
/// and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyTargetTransformInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

/// Attempt to convert [relaxed_]swizzle to shufflevector if the mask is
/// constant.
static Value *simplifyWasmSwizzle(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder,
                                  bool IsRelaxed) {
  auto *V = dyn_cast<Constant>(II.getArgOperand(1));
  if (!V)
    return nullptr;

  auto *VecTy = cast<FixedVectorType>(II.getType());
  unsigned NumElts = VecTy->getNumElements();
  assert(NumElts == 16);

  // Construct a shuffle mask from constant integers or UNDEFs.
  int Indexes[16];
  bool AnyOutOfBounds = false;

  for (unsigned I = 0; I < NumElts; ++I) {
    Constant *COp = V->getAggregateElement(I);
    if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp)))
      return nullptr;

    if (isa<UndefValue>(COp)) {
      Indexes[I] = -1;
      continue;
    }

    int64_t Index = cast<ConstantInt>(COp)->getSExtValue();

    if (Index >= NumElts && IsRelaxed) {
      // For lane indices above 15, the relaxed_swizzle operation can choose
      // between returning 0 or the lane at `Index % 16`. However, the choice
      // must be made consistently. As the WebAssembly spec states:
      //
      // "The result of relaxed operators are implementation-dependent, because
      // the set of possible results may depend on properties of the host
      // environment, such as its hardware. Technically, their behaviour is
      // controlled by a set of global parameters to the semantics that an
      // implementation can instantiate in different ways. These choices are
      // fixed, that is, parameters are constant during the execution of any
      // given program."
      //
      // The WebAssembly runtime may choose differently from us, so we can't
      // optimize a relaxed swizzle with lane indices above 15.
      return nullptr;
    }

    if (Index >= NumElts || Index < 0) {
      AnyOutOfBounds = true;
      // If there are out-of-bounds indices, the swizzle instruction returns
      // zeroes in those lanes. We'll provide an all-zeroes vector as the
      // second argument to shufflevector and read the first element from it.
      Indexes[I] = NumElts;
      continue;
    }

    Indexes[I] = Index;
  }

  auto *V1 = II.getArgOperand(0);
  auto *V2 =
      AnyOutOfBounds ? Constant::getNullValue(VecTy) : PoisonValue::get(VecTy);

  return Builder.CreateShuffleVector(V1, V2, ArrayRef(Indexes, NumElts));
}

std::optional<Instruction *>
WebAssemblyTTIImpl::instCombineIntrinsic(InstCombiner &IC,
                                         IntrinsicInst &II) const {
  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  case Intrinsic::wasm_swizzle:
  case Intrinsic::wasm_relaxed_swizzle:
    if (Value *V = simplifyWasmSwizzle(
            II, IC.Builder, IID == Intrinsic::wasm_relaxed_swizzle)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;
  }

  return std::nullopt;
}
