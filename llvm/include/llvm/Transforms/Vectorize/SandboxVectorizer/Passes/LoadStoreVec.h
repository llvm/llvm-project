//===- LoadStoreVec.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that vectorizes short store-load chains.
// Unlike generic bundle vectorization, this pass can vectorize instructions
// of different types.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_LOADSTOREVEC_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_LOADSTOREVEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/SandboxIR/Pass.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

#include <optional>

namespace llvm {

class DataLayout;

namespace sandboxir {

class Context;
class Function;
class Value;
class Instruction;
class Type;
class Constant;

/// Reinterprets the bits of a value as another type of the same size, e.g.,
/// float to i32, ptr to i64, or double to <2 x i32>.
class Reinterpret {
  /// A single cast in the chain built by getCastChain().
  using CastStep = std::pair<Instruction::Opcode, Type *>;

  /// \returns an integer type with the same layout as \p Ty: iN for scalars,
  /// <N x iM> for vectors.
  static Type *getIntTypeFor(Type *Ty, const DataLayout &DL);

  /// \returns the casts that reinterpret \p SrcTy as \p DestTy, empty if the
  /// types match, or nullopt if a non-integral pointer is involved.
  static std::optional<SmallVector<CastStep, 3>>
  getCastChain(Type *SrcTy, Type *DestTy, const DataLayout &DL);

public:
  /// \returns true if \p SrcTy can be reinterpreted as \p DestTy.
  static bool canReinterpret(Type *SrcTy, Type *DestTy, const DataLayout &DL) {
    return getCastChain(SrcTy, DestTy, DL).has_value();
  }

  /// Reinterprets \p V as \p DestTy, which must have the same size in \p DL,
  /// inserting casts before \p WhereIt. Casts of constants fold, so nothing is
  /// inserted for those. \returns nullptr if a non-integral pointer is
  /// involved.
  static Value *reinterpretValue(Value *V, Type *DestTy,
                                 BasicBlock::iterator WhereIt, Context &Ctx,
                                 const DataLayout &DL);

  /// Reinterprets \p C as \p DestTy, e.g., float 1.0 to i32 0x3f800000, or a
  /// null pointer to i64 0. \returns nullptr if a non-integral pointer is
  /// involved.
  static Constant *reinterpretConstant(Constant *C, Type *DestTy,
                                       const DataLayout &DL);
};

class LLVM_ABI LoadStoreVec final : public RegionPass {
  const DataLayout *DL = nullptr;
  /// The region saved by saveIR(), used by acceptOrRevert().
  Region *Rgn = nullptr;
  /// The region's cost at the time of saveIR().
  InstructionCost CostBefore = 0;
  const Analyses *A = nullptr;
  Context *Ctx = nullptr;
  std::unique_ptr<LegalityAnalysis> Legality;
  VecUtils::DeadInstructionMorgue DeadInstrMorgue;

  /// Saves the IR along with the current cost of \p Rgn, so that
  /// acceptOrRevert() can tell whether vectorizing was profitable.
  void saveIR(Region &Rgn);

  /// Accepts the transaction saved by saveIR() if vectorizing was profitable,
  /// reverts it otherwise. \returns true if the transaction was accepted.
  bool acceptOrRevert();

  /// Checks legality of vectorization and \returns the vector type on success,
  /// nullopt otherwise.
  std::optional<Type *> canVectorize(BndlRef<Instruction *> Bndl);

  /// Builds a single vector load out of \p Loads. \returns the new load,
  /// or nullptr if \p Loads are not a vectorizable.
  LoadInst *createVectorLoad(BndlRef<Instruction *> Loads);

  /// Packs \p Constants into one constant vector. The element type is chosen
  /// by VecUtils::getCombinedVectorTypeFor().
  ///
  /// \returns the packed constant vector, or nullptr if some constant cannot
  /// be converted, e.g., a vector constant that is not yet flattened, a
  /// non-integral pointer, or a wide constant without a known bit pattern such
  /// as a ConstantExpr or a global's address.
  Constant *createConstantVector(BndlRef<Constant *> Constants);

  /// Vectorizes \p Stores and their operands if constants or consecutive
  /// loads. \returns true on success.
  bool vectorizeStores(BndlRef<Instruction *> Stores, Region &Rgn);

  /// Vectorizes \p Loads into a single load. \return the packed load.
  LoadInst *vectorizeLoads(BndlRef<Instruction *> Loads, Region &Rgn);

public:
  LoadStoreVec(StringRef AuxArg) : RegionPass("load-store-vec") {
    assert(AuxArg.empty() && "This pass ignores aux arg!");
  }
  bool runOnRegion(Region &Rgn, const Analyses &A) final;
};

} // namespace sandboxir

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_STRUCTINITVEC_H
