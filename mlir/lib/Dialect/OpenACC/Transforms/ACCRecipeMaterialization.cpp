//===- ACCRecipeMaterialization.cpp - Materialize ACC recipes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Overview:
// ---------
// OpenACC compute constructs (acc.parallel, acc.serial, acc.kernels) and
// acc.loop can carry data clauses (acc.private, acc.firstprivate,
// acc.reduction) that refer to recipes (acc.private.recipe,
// acc.firstprivate.recipe, acc.reduction.recipe). Recipes define how to
// initialize, copy, combine, or destroy a particular variable. This pass clones
// those regions into the construct and ensures the materialized SSA values are
// used instead.
//
// Transforms:
// -----------
// 1. Firstprivate: Inserts acc.firstprivate_map so the initial value is
//    available on the device, then clones the recipe init and copy regions
//    into the construct and replaces uses with the materialized alloca.
//    Optional destroy region is cloned before the region terminator.
//
// 2. Private: Clones the recipe init region into the construct (at the
//    region entry or at the loop op for acc.loop private). Replaces uses
//    of the recipe result with the materialized alloca. Optional destroy
//    region is cloned before the region terminator.
//
// 3. Reduction: Creates acc.reduction_init (init region inlined) and
//    acc.reduction_combine_region (combiner region inlined). Uses within
//    the region are updated to the reduction init result.
//
// Requirements:
// -------------
// 1. OpenACCSupport: The pass uses the `acc::OpenACCSupport` analysis
//    including emitNYI for unsupported cases.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsLoop.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCRECIPEMATERIALIZATION
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-recipe-materialization"

namespace {

using namespace mlir;

static void saveVarName(StringRef name, Value dst) {
  if (name.empty())
    return;
  if (Operation *dstOp = dst.getDefiningOp()) {
    if (dstOp->getAttrOfType<acc::VarNameAttr>(acc::getVarNameAttrName()))
      return;
    if (isa<ACC_DATA_ENTRY_OPS>(dstOp))
      return;
    dstOp->setAttr(acc::getVarNameAttrName(),
                   acc::VarNameAttr::get(dstOp->getContext(), name));
    return;
  }
  auto blockArg = dyn_cast<BlockArgument>(dst);
  if (!blockArg)
    return;
  Block *block = blockArg.getOwner();
  Region *region = block ? block->getParent() : nullptr;
  if (!region || !block->isEntryBlock())
    return;
  Operation *parent = region->getParentOp();
  if (!parent)
    return;
  auto funcOp = dyn_cast<FunctionOpInterface>(parent);
  if (!funcOp)
    return;
  unsigned argIdx = blockArg.getArgNumber();
  if (argIdx >= funcOp.getNumArguments())
    return;
  if (funcOp.getArgAttr(argIdx, acc::getVarNameAttrName()))
    return;
  funcOp.setArgAttr(argIdx, acc::getVarNameAttrName(),
                    acc::VarNameAttr::get(parent->getContext(), name));
}

static void saveVarName(Value src, Value dst) {
  saveVarName(acc::getVariableName(src), dst);
}

// Clone the destroy region of the recipe before the terminator of the provided
// block. Values must be provided for the destroy region block arguments
// according to the recipe specifications.
template <typename RecipeOpTy>
static void cloneDestroy(RecipeOpTy recipe, mlir::Block *block,
                         const llvm::SmallVector<mlir::Value> &arguments) {
  IRMapping mapping{};
  Region &destroyRegion = recipe.getDestroyRegion();
  assert(destroyRegion.getBlocks().front().getNumArguments() ==
             arguments.size() &&
         "unexpected acc recipe destroy block arguments");
  mapping.map(destroyRegion.getBlocks().front().getArguments(), arguments);
  acc::cloneACCRegionInto(&destroyRegion, block, std::prev(block->end()),
                          mapping,
                          /*resultsToReplace=*/{});
}

class ACCRecipeMaterialization
    : public acc::impl::ACCRecipeMaterializationBase<ACCRecipeMaterialization> {
public:
  using acc::impl::ACCRecipeMaterializationBase<
      ACCRecipeMaterialization>::ACCRecipeMaterializationBase;
  void runOnOperation() override;

private:
  // When handling firstprivate, the initial value needs to be available on
  // the GPU. One way to get that value there is to map the variable through
  // global memory.
  // Thus, when we materialize a firstprivate, we materialize it into
  // a mapping action first. This function ends up with doing the following:
  // %dev = acc.firstprivate var(%var)
  // =>
  // %copy = acc.firstprivate_map var(%var)
  // %dev = acc.firstprivate var(%copy)
  // When the recipe materialization happens, the `acc.firstprivate` ends up
  // being removed. But because of the way we chain it to the
  // `acc.firstprivate_map`, then its result becomes live-in to the
  // compute region and used as the variable the initial value is loaded from.
  void handleFirstprivateMapping(acc::FirstprivateOp firstprivateOp) const;
  template <typename OpTy>
  void removeRecipe(OpTy op, ModuleOp moduleOp) const;
  template <typename OpTy, typename RecipeOpTy, typename AccOpTy>
  LogicalResult materialize(OpTy op, RecipeOpTy recipe, AccOpTy accOp,
                            acc::OpenACCSupport &accSupport) const;
  template <typename OpTy>
  LogicalResult materializeForACCOp(OpTy accOp,
                                    acc::OpenACCSupport &accSupport) const;
};

void ACCRecipeMaterialization::handleFirstprivateMapping(
    acc::FirstprivateOp firstprivateOp) const {
  OpBuilder builder(firstprivateOp);
  auto mapFirstprivateOp = acc::FirstprivateMapInitialOp::create(
      builder, firstprivateOp.getLoc(), firstprivateOp.getVar(),
      firstprivateOp.getStructured(), firstprivateOp.getImplicit(),
      firstprivateOp.getBounds());
  mapFirstprivateOp.setName(firstprivateOp.getName());
  firstprivateOp.getVarMutable().assign(mapFirstprivateOp.getAccVar());
}

template <typename OpTy>
void ACCRecipeMaterialization::removeRecipe(OpTy op, ModuleOp moduleOp) const {
  auto recipeName = op.getNameAttr();
  if (SymbolTable::symbolKnownUseEmpty(recipeName, moduleOp)) {
    LLVM_DEBUG(llvm::dbgs() << "erasing recipe: " << recipeName << "\n");
    op.erase();
  } else {
    LLVM_DEBUG({
      std::optional<SymbolTable::UseRange> symbolUses =
          op.getSymbolUses(moduleOp);
      if (symbolUses.has_value()) {
        for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
          llvm::dbgs() << "symbol use: ";
          symbolUse.getUser()->dump();
        }
      }
    });
    llvm_unreachable("expected no use of recipe symbol");
  }
}

template <typename OpTy, typename RecipeOpTy, typename AccOpTy>
LogicalResult
ACCRecipeMaterialization::materialize(OpTy op, RecipeOpTy recipe, AccOpTy accOp,
                                      acc::OpenACCSupport &accSupport) const {
  Region &region = accOp.getRegion();
  Value origPtr = op.getVar();
  Value accPtr = op.getAccVar();
  assert(accPtr && "invalid op: null acc var");

  OpBuilder b(op);
  SmallVector<Value> triples;

  // Clone init block into the region at the insertion point specified.
  Region &initRegion = recipe.getInitRegion();
  unsigned initNumArguments =
      initRegion.getBlocks().front().getArguments().size();
  if (initNumArguments > 1) {
    // Code from C/C++ will most likely only provide extent arguments to the
    // recipe arguments.
    if ((initNumArguments - 1) % 3 != 0) {
      (void)accSupport.emitNYI(recipe.getLoc(),
                               "privatization of array section with extents");
      return failure();
    }
    // The remaining arguments must be the bounds triples
    // (lower-bound, upper-bound, step), ...
    unsigned argIdx = 1;
    // Cast the given value to the type of the combiner region's argument
    // at position argIdx, and increment argIdx.
    auto castValueToArgType = [&](Location loc, Value v) {
      return convertScalarToDtype(
          b, loc, v,
          initRegion.getBlocks().front().getArgument(argIdx++).getType(),
          /*isUnsignedCast=*/false);
    };
    for (Value bound : acc::getBounds(op)) {
      auto dataBound = bound.getDefiningOp<acc::DataBoundsOp>();
      assert(dataBound &&
             "acc.reduction's bound must be defined by acc.bounds");
      // NOTE: we should probably generate get_lowerbound, get_upperbound
      // and get_stride here, so that we can stop looking for the acc.bounds
      // operation above, and just use the `bound` value.
      Value lb =
          castValueToArgType(dataBound.getLoc(), dataBound.getLowerbound());
      Value ub =
          castValueToArgType(dataBound.getLoc(), dataBound.getUpperbound());
      Value step =
          castValueToArgType(dataBound.getLoc(), dataBound.getStride());
      triples.append({lb, ub, step});
    }
    assert(triples.size() + 1 == initNumArguments &&
           "mismatch between number bounds and number of recipe init block "
           "arguments");
  }

  IRMapping mapping;
  SmallVector<Value> initArgs{origPtr};
  initArgs.append(triples);
  mapping.map(initRegion.getBlocks().front().getArguments(), initArgs);

  if constexpr (std::is_same_v<OpTy, acc::PrivateOp>) {
    // Clone the init region for a private.
    Block *block = &region.front();
    auto [results, ip] = acc::cloneACCRegionInto(
        &initRegion, block, block->begin(), mapping, {accPtr});
    assert(results.size() == 1 && "expected single result from init region");
    saveVarName(op.getAccVar(), results[0]);
    // Clone the destroy region for a private, if it exists.
    if (!recipe.getDestroyRegion().empty()) {
      results.insert(results.begin(), origPtr);
      results.append(triples);
      cloneDestroy(recipe, block, results);
    }
  } else if constexpr (std::is_same_v<OpTy, acc::FirstprivateOp>) {
    // Clone the init region for a firstprivate.
    Block *block = &region.front();
    auto [results, ip] = acc::cloneACCRegionInto(
        &initRegion, block, block->begin(), mapping, {accPtr});
    assert(results.size() == 1 && "expected single result from init region");
    saveVarName(op.getAccVar(), results[0]);
    // We want the copy to store the origPtr to private
    results.insert(results.begin(), origPtr);
    results.append(triples);

    // Clone the copy region for a firstprivate
    mapping.clear();
    mapping.map(recipe.getCopyRegion().front().getArguments(), results);
    // Clone the copy region for a firstprivate.
    acc::cloneACCRegionInto(&recipe.getCopyRegion(), block, std::next(ip),
                            mapping, {});
    if (!recipe.getDestroyRegion().empty()) {
      // origPtr was already pushed.
      cloneDestroy(recipe, block, results);
    }
  } else if constexpr (std::is_same_v<OpTy, acc::ReductionOp>) {
    auto cloneRegionIntoAccRegion = [&](Region *src, Region *dest,
                                        bool hasResult) {
      src->cloneInto(dest, mapping);
      Block *block = &dest->front();
      Operation *terminator = block->getTerminator();
      b.setInsertionPoint(terminator);
      if (hasResult)
        acc::YieldOp::create(b, op.getLoc(), terminator->getOperands());
      else
        acc::YieldOp::create(b, op.getLoc(), ValueRange{});
      terminator->erase();
    };

    // Clone the init region into acc.reduction_init.
    if constexpr (std::is_same_v<AccOpTy, acc::ParallelOp>)
      b.setInsertionPointToStart(&region.front());
    else if constexpr (std::is_same_v<AccOpTy, acc::LoopOp>)
      b.setInsertionPoint(op);
    else
      llvm_unreachable("unexpected acc op with reduction recipe");

    auto reductionOp = acc::ReductionInitOp::create(
        b, op.getLoc(), origPtr, recipe.getReductionOperatorAttr());
    saveVarName(op.getAccVar(), reductionOp.getResult());
    cloneRegionIntoAccRegion(&initRegion, &reductionOp.getRegion(),
                             /*hasResult=*/true);

    // Update the uses within the loop to use the reduction op result.
    replaceAllUsesInRegionWith(accPtr, reductionOp.getResult(), region);

    // Clone the combiner region into acc.reduction_combine_region.
    Region &combinerRegion = recipe.getCombinerRegion();
    Block *entryBlock = &combinerRegion.front();

    if constexpr (std::is_same_v<AccOpTy, acc::ParallelOp>)
      b.setInsertionPoint(region.back().getTerminator());
    else if constexpr (std::is_same_v<AccOpTy, acc::LoopOp>)
      b.setInsertionPointAfter(accOp);
    else
      llvm_unreachable("unexpected acc op with reduction recipe");

    // Map the first two block arguments to the original and private
    // reduction variables. If the recipe's combiner region has the bounds
    // arguments, we have to map them to the corresponding operands of
    // acc.reduction operation.
    mapping.clear();
    SmallVector<Value, 2> argsRemapping{origPtr, reductionOp.getResult()};
    argsRemapping.append(triples);
    mapping.map(entryBlock->getArguments(), argsRemapping);

    auto combineRegionOp = acc::ReductionCombineRegionOp::create(
        b, op.getLoc(), origPtr, reductionOp.getResult());
    cloneRegionIntoAccRegion(&combinerRegion, &combineRegionOp.getRegion(),
                             /*hasResult=*/false);

    auto setSeqParDimsForRecipeLoops = [](Region *r) {
      r->walk([](LoopLikeOpInterface loopLike) {
        loopLike->setAttr(
            acc::GPUParallelDimsAttr::name,
            acc::GPUParallelDimsAttr::seq(loopLike->getContext()));
      });
    };
    setSeqParDimsForRecipeLoops(&reductionOp.getRegion());
    setSeqParDimsForRecipeLoops(&combineRegionOp.getRegion());

    if (!recipe.getDestroyRegion().empty()) {
      (void)accSupport.emitNYI(
          recipe.getLoc(),
          "OpenACC reduction variable that requires destruction code");
      return failure();
    }
  } else {
    llvm_unreachable("unexpected op type");
  }

  op.erase();
  return success();
}

template <typename OpTy>
LogicalResult ACCRecipeMaterialization::materializeForACCOp(
    OpTy accOp, acc::OpenACCSupport &accSupport) const {
  assert(isa<ACC_COMPUTE_CONSTRUCT_AND_LOOP_OPS>(accOp));

  if (!accOp.getFirstprivateOperands().empty()) {
    // Clear the firstprivate operands list so there will be no uses after
    // the recipe is materialized.
    SmallVector<Value> operands(accOp.getFirstprivateOperands());
    accOp.getFirstprivateOperandsMutable().clear();
    for (Value operand : operands) {
      auto firstprivateOp = cast<acc::FirstprivateOp>(operand.getDefiningOp());
      auto symbolRef = cast<SymbolRefAttr>(firstprivateOp.getRecipeAttr());
      auto decl = SymbolTable::lookupNearestSymbolFrom(accOp, symbolRef);
      auto recipeOp = cast<acc::FirstprivateRecipeOp>(decl);
      LLVM_DEBUG(llvm::dbgs() << "materializing: " << firstprivateOp << "\n"
                              << symbolRef << "\n");
      handleFirstprivateMapping(firstprivateOp);
      if (failed(materialize(firstprivateOp, recipeOp, accOp, accSupport)))
        return failure();
    }
  }

  if (!accOp.getPrivateOperands().empty()) {
    // Clear the private operands list so there will be no uses after
    // the recipe is materialized.
    SmallVector<Value> operands(accOp.getPrivateOperands());
    accOp.getPrivateOperandsMutable().clear();
    for (Value operand : operands) {
      auto privateOp = cast<acc::PrivateOp>(operand.getDefiningOp());
      auto symbolRef = cast<SymbolRefAttr>(privateOp.getRecipeAttr());
      auto decl = SymbolTable::lookupNearestSymbolFrom(accOp, symbolRef);
      auto recipeOp = cast<acc::PrivateRecipeOp>(decl);
      LLVM_DEBUG(llvm::dbgs() << "materializing: " << privateOp << "\n"
                              << symbolRef << "\n");
      if (failed(materialize(privateOp, recipeOp, accOp, accSupport)))
        return failure();
    }
  }

  if (!accOp.getReductionOperands().empty()) {
    // Clear the reduction operands list so there will be no uses after
    // the recipe is materialized.
    SmallVector<Value> operands(accOp.getReductionOperands());
    accOp.getReductionOperandsMutable().clear();
    for (Value operand : operands) {
      auto reductionOp = cast<acc::ReductionOp>(operand.getDefiningOp());
      auto symbolRef = cast<SymbolRefAttr>(reductionOp.getRecipeAttr());
      auto decl = SymbolTable::lookupNearestSymbolFrom(accOp, symbolRef);
      auto recipeOp = cast<acc::ReductionRecipeOp>(decl);
      LLVM_DEBUG(llvm::dbgs() << "materializing: " << reductionOp << "\n"
                              << symbolRef << "\n");
      if (failed(materialize(reductionOp, recipeOp, accOp, accSupport)))
        return failure();
    }
  }
  return success();
}

void ACCRecipeMaterialization::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  acc::OpenACCSupport &accSupport = getAnalysis<acc::OpenACCSupport>();

  // Materialize all recipes for all compute constructs and loop constructs.
  bool anyFailed = false;
  moduleOp.walk([&](Operation *op) {
    if (anyFailed)
      return;
    TypeSwitch<Operation *>(op).Case<ACC_COMPUTE_CONSTRUCT_AND_LOOP_OPS>(
        [&](auto constructOp) {
          if (failed(materializeForACCOp(constructOp, accSupport)))
            anyFailed = true;
        });
  });
  if (anyFailed) {
    signalPassFailure();
    return;
  }

  // Remove all recipes.
  moduleOp.walk([&](Operation *op) {
    if (auto recipe = dyn_cast<acc::ReductionRecipeOp>(op))
      removeRecipe(recipe, moduleOp);
    else if (auto recipe = dyn_cast<acc::PrivateRecipeOp>(op))
      removeRecipe(recipe, moduleOp);
    else if (auto recipe = dyn_cast<acc::FirstprivateRecipeOp>(op))
      removeRecipe(recipe, moduleOp);
  });
}

} // namespace
