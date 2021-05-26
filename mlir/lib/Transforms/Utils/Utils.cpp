//===- Utils.cpp ---- Misc utilities for code and data transformation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous transformation routines for non-loop IR
// structures.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Utils.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;

// Perform the replacement in `op`.
LogicalResult mlir::replaceAllMemRefUsesWith(Value oldMemRef, Value newMemRef,
                                             Operation *op,
                                             ArrayRef<Value> extraIndices,
                                             AffineMap indexRemap,
                                             ArrayRef<Value> extraOperands,
                                             ArrayRef<Value> symbolOperands,
                                             bool allowNonDereferencingOps) {
  unsigned newMemRefRank = newMemRef.getType().cast<MemRefType>().getRank();
  (void)newMemRefRank; // unused in opt mode
  unsigned oldMemRefRank = oldMemRef.getType().cast<MemRefType>().getRank();
  (void)oldMemRefRank; // unused in opt mode
  if (indexRemap) {
    assert(indexRemap.getNumSymbols() == symbolOperands.size() &&
           "symbolic operand count mismatch");
    assert(indexRemap.getNumInputs() ==
           extraOperands.size() + oldMemRefRank + symbolOperands.size());
    assert(indexRemap.getNumResults() + extraIndices.size() == newMemRefRank);
  } else {
    assert(oldMemRefRank + extraIndices.size() == newMemRefRank);
  }

  // Assert same elemental type.
  assert(oldMemRef.getType().cast<MemRefType>().getElementType() ==
         newMemRef.getType().cast<MemRefType>().getElementType());

  SmallVector<unsigned, 2> usePositions;
  for (const auto &opEntry : llvm::enumerate(op->getOperands())) {
    if (opEntry.value() == oldMemRef)
      usePositions.push_back(opEntry.index());
  }

  // If memref doesn't appear, nothing to do.
  if (usePositions.empty())
    return success();

  if (usePositions.size() > 1) {
    // TODO: extend it for this case when needed (rare).
    assert(false && "multiple dereferencing uses in a single op not supported");
    return failure();
  }

  unsigned memRefOperandPos = usePositions.front();

  OpBuilder builder(op);
  // The following checks if op is dereferencing memref and performs the access
  // index rewrites.
  auto affMapAccInterface = dyn_cast<AffineMapAccessInterface>(op);
  if (!affMapAccInterface) {
    if (!allowNonDereferencingOps) {
      // Failure: memref used in a non-dereferencing context (potentially
      // escapes); no replacement in these cases unless allowNonDereferencingOps
      // is set.
      return failure();
    }
    op->setOperand(memRefOperandPos, newMemRef);
    return success();
  }
  // Perform index rewrites for the dereferencing op and then replace the op
  NamedAttribute oldMapAttrPair =
      affMapAccInterface.getAffineMapAttrForMemRef(oldMemRef);
  AffineMap oldMap = oldMapAttrPair.second.cast<AffineMapAttr>().getValue();
  unsigned oldMapNumInputs = oldMap.getNumInputs();
  SmallVector<Value, 4> oldMapOperands(
      op->operand_begin() + memRefOperandPos + 1,
      op->operand_begin() + memRefOperandPos + 1 + oldMapNumInputs);

  // Apply 'oldMemRefOperands = oldMap(oldMapOperands)'.
  SmallVector<Value, 4> oldMemRefOperands;
  SmallVector<Value, 4> affineApplyOps;
  oldMemRefOperands.reserve(oldMemRefRank);
  if (oldMap != builder.getMultiDimIdentityMap(oldMap.getNumDims())) {
    for (auto resultExpr : oldMap.getResults()) {
      auto singleResMap = AffineMap::get(oldMap.getNumDims(),
                                         oldMap.getNumSymbols(), resultExpr);
      auto afOp = builder.create<AffineApplyOp>(op->getLoc(), singleResMap,
                                                oldMapOperands);
      oldMemRefOperands.push_back(afOp);
      affineApplyOps.push_back(afOp);
    }
  } else {
    oldMemRefOperands.assign(oldMapOperands.begin(), oldMapOperands.end());
  }

  // Construct new indices as a remap of the old ones if a remapping has been
  // provided. The indices of a memref come right after it, i.e.,
  // at position memRefOperandPos + 1.
  SmallVector<Value, 4> remapOperands;
  remapOperands.reserve(extraOperands.size() + oldMemRefRank +
                        symbolOperands.size());
  remapOperands.append(extraOperands.begin(), extraOperands.end());
  remapOperands.append(oldMemRefOperands.begin(), oldMemRefOperands.end());
  remapOperands.append(symbolOperands.begin(), symbolOperands.end());

  SmallVector<Value, 4> remapOutputs;
  remapOutputs.reserve(oldMemRefRank);

  if (indexRemap &&
      indexRemap != builder.getMultiDimIdentityMap(indexRemap.getNumDims())) {
    // Remapped indices.
    for (auto resultExpr : indexRemap.getResults()) {
      auto singleResMap = AffineMap::get(
          indexRemap.getNumDims(), indexRemap.getNumSymbols(), resultExpr);
      auto afOp = builder.create<AffineApplyOp>(op->getLoc(), singleResMap,
                                                remapOperands);
      remapOutputs.push_back(afOp);
      affineApplyOps.push_back(afOp);
    }
  } else {
    // No remapping specified.
    remapOutputs.assign(remapOperands.begin(), remapOperands.end());
  }

  SmallVector<Value, 4> newMapOperands;
  newMapOperands.reserve(newMemRefRank);

  // Prepend 'extraIndices' in 'newMapOperands'.
  for (Value extraIndex : extraIndices) {
    assert(extraIndex.getDefiningOp()->getNumResults() == 1 &&
           "single result op's expected to generate these indices");
    assert((isValidDim(extraIndex) || isValidSymbol(extraIndex)) &&
           "invalid memory op index");
    newMapOperands.push_back(extraIndex);
  }

  // Append 'remapOutputs' to 'newMapOperands'.
  newMapOperands.append(remapOutputs.begin(), remapOutputs.end());

  // Create new fully composed AffineMap for new op to be created.
  assert(newMapOperands.size() == newMemRefRank);
  auto newMap = builder.getMultiDimIdentityMap(newMemRefRank);
  // TODO: Avoid creating/deleting temporary AffineApplyOps here.
  fullyComposeAffineMapAndOperands(&newMap, &newMapOperands);
  newMap = simplifyAffineMap(newMap);
  canonicalizeMapAndOperands(&newMap, &newMapOperands);
  // Remove any affine.apply's that became dead as a result of composition.
  for (Value value : affineApplyOps)
    if (value.use_empty())
      value.getDefiningOp()->erase();

  OperationState state(op->getLoc(), op->getName());
  // Construct the new operation using this memref.
  state.operands.reserve(op->getNumOperands() + extraIndices.size());
  // Insert the non-memref operands.
  state.operands.append(op->operand_begin(),
                        op->operand_begin() + memRefOperandPos);
  // Insert the new memref value.
  state.operands.push_back(newMemRef);

  // Insert the new memref map operands.
  state.operands.append(newMapOperands.begin(), newMapOperands.end());

  // Insert the remaining operands unmodified.
  state.operands.append(op->operand_begin() + memRefOperandPos + 1 +
                            oldMapNumInputs,
                        op->operand_end());

  // Result types don't change. Both memref's are of the same elemental type.
  state.types.reserve(op->getNumResults());
  for (auto result : op->getResults())
    state.types.push_back(result.getType());

  // Add attribute for 'newMap', other Attributes do not change.
  auto newMapAttr = AffineMapAttr::get(newMap);
  for (auto namedAttr : op->getAttrs()) {
    if (namedAttr.first == oldMapAttrPair.first)
      state.attributes.push_back({namedAttr.first, newMapAttr});
    else
      state.attributes.push_back(namedAttr);
  }

  // Create the new operation.
  auto *repOp = builder.createOperation(state);
  op->replaceAllUsesWith(repOp);
  op->erase();

  return success();
}

LogicalResult mlir::replaceAllMemRefUsesWith(
    Value oldMemRef, Value newMemRef, ArrayRef<Value> extraIndices,
    AffineMap indexRemap, ArrayRef<Value> extraOperands,
    ArrayRef<Value> symbolOperands, Operation *domInstFilter,
    Operation *postDomInstFilter, bool allowNonDereferencingOps,
    bool replaceInDeallocOp) {
  unsigned newMemRefRank = newMemRef.getType().cast<MemRefType>().getRank();
  (void)newMemRefRank; // unused in opt mode
  unsigned oldMemRefRank = oldMemRef.getType().cast<MemRefType>().getRank();
  (void)oldMemRefRank;
  if (indexRemap) {
    assert(indexRemap.getNumSymbols() == symbolOperands.size() &&
           "symbol operand count mismatch");
    assert(indexRemap.getNumInputs() ==
           extraOperands.size() + oldMemRefRank + symbolOperands.size());
    assert(indexRemap.getNumResults() + extraIndices.size() == newMemRefRank);
  } else {
    assert(oldMemRefRank + extraIndices.size() == newMemRefRank);
  }

  // Assert same elemental type.
  assert(oldMemRef.getType().cast<MemRefType>().getElementType() ==
         newMemRef.getType().cast<MemRefType>().getElementType());

  std::unique_ptr<DominanceInfo> domInfo;
  std::unique_ptr<PostDominanceInfo> postDomInfo;
  if (domInstFilter)
    domInfo = std::make_unique<DominanceInfo>(
        domInstFilter->getParentOfType<FuncOp>());

  if (postDomInstFilter)
    postDomInfo = std::make_unique<PostDominanceInfo>(
        postDomInstFilter->getParentOfType<FuncOp>());

  // Walk all uses of old memref; collect ops to perform replacement. We use a
  // DenseSet since an operation could potentially have multiple uses of a
  // memref (although rare), and the replacement later is going to erase ops.
  DenseSet<Operation *> opsToReplace;
  for (auto *op : oldMemRef.getUsers()) {
    // Skip this use if it's not dominated by domInstFilter.
    if (domInstFilter && !domInfo->dominates(domInstFilter, op))
      continue;

    // Skip this use if it's not post-dominated by postDomInstFilter.
    if (postDomInstFilter && !postDomInfo->postDominates(postDomInstFilter, op))
      continue;

    // Skip dealloc's - no replacement is necessary, and a memref replacement
    // at other uses doesn't hurt these dealloc's.
    if (isa<memref::DeallocOp>(op) && !replaceInDeallocOp)
      continue;

    // Check if the memref was used in a non-dereferencing context. It is fine
    // for the memref to be used in a non-dereferencing way outside of the
    // region where this replacement is happening.
    if (!isa<AffineMapAccessInterface>(*op)) {
      if (!allowNonDereferencingOps)
        return failure();
      // Currently we support the following non-dereferencing ops to be a
      // candidate for replacement: Dealloc, CallOp and ReturnOp.
      // TODO: Add support for other kinds of ops.
      if (!op->hasTrait<OpTrait::MemRefsNormalizable>())
        return failure();
    }

    // We'll first collect and then replace --- since replacement erases the op
    // that has the use, and that op could be postDomFilter or domFilter itself!
    opsToReplace.insert(op);
  }

  for (auto *op : opsToReplace) {
    if (failed(replaceAllMemRefUsesWith(
            oldMemRef, newMemRef, op, extraIndices, indexRemap, extraOperands,
            symbolOperands, allowNonDereferencingOps)))
      llvm_unreachable("memref replacement guaranteed to succeed here");
  }

  return success();
}

/// Given an operation, inserts one or more single result affine
/// apply operations, results of which are exclusively used by this operation
/// operation. The operands of these newly created affine apply ops are
/// guaranteed to be loop iterators or terminal symbols of a function.
///
/// Before
///
/// affine.for %i = 0 to #map(%N)
///   %idx = affine.apply (d0) -> (d0 mod 2) (%i)
///   "send"(%idx, %A, ...)
///   "compute"(%idx)
///
/// After
///
/// affine.for %i = 0 to #map(%N)
///   %idx = affine.apply (d0) -> (d0 mod 2) (%i)
///   "send"(%idx, %A, ...)
///   %idx_ = affine.apply (d0) -> (d0 mod 2) (%i)
///   "compute"(%idx_)
///
/// This allows applying different transformations on send and compute (for eg.
/// different shifts/delays).
///
/// Returns nullptr either if none of opInst's operands were the result of an
/// affine.apply and thus there was no affine computation slice to create, or if
/// all the affine.apply op's supplying operands to this opInst did not have any
/// uses besides this opInst; otherwise returns the list of affine.apply
/// operations created in output argument `sliceOps`.
void mlir::createAffineComputationSlice(
    Operation *opInst, SmallVectorImpl<AffineApplyOp> *sliceOps) {
  // Collect all operands that are results of affine apply ops.
  SmallVector<Value, 4> subOperands;
  subOperands.reserve(opInst->getNumOperands());
  for (auto operand : opInst->getOperands())
    if (isa_and_nonnull<AffineApplyOp>(operand.getDefiningOp()))
      subOperands.push_back(operand);

  // Gather sequence of AffineApplyOps reachable from 'subOperands'.
  SmallVector<Operation *, 4> affineApplyOps;
  getReachableAffineApplyOps(subOperands, affineApplyOps);
  // Skip transforming if there are no affine maps to compose.
  if (affineApplyOps.empty())
    return;

  // Check if all uses of the affine apply op's lie only in this op op, in
  // which case there would be nothing to do.
  bool localized = true;
  for (auto *op : affineApplyOps) {
    for (auto result : op->getResults()) {
      for (auto *user : result.getUsers()) {
        if (user != opInst) {
          localized = false;
          break;
        }
      }
    }
  }
  if (localized)
    return;

  OpBuilder builder(opInst);
  SmallVector<Value, 4> composedOpOperands(subOperands);
  auto composedMap = builder.getMultiDimIdentityMap(composedOpOperands.size());
  fullyComposeAffineMapAndOperands(&composedMap, &composedOpOperands);

  // Create an affine.apply for each of the map results.
  sliceOps->reserve(composedMap.getNumResults());
  for (auto resultExpr : composedMap.getResults()) {
    auto singleResMap = AffineMap::get(composedMap.getNumDims(),
                                       composedMap.getNumSymbols(), resultExpr);
    sliceOps->push_back(builder.create<AffineApplyOp>(
        opInst->getLoc(), singleResMap, composedOpOperands));
  }

  // Construct the new operands that include the results from the composed
  // affine apply op above instead of existing ones (subOperands). So, they
  // differ from opInst's operands only for those operands in 'subOperands', for
  // which they will be replaced by the corresponding one from 'sliceOps'.
  SmallVector<Value, 4> newOperands(opInst->getOperands());
  for (unsigned i = 0, e = newOperands.size(); i < e; i++) {
    // Replace the subOperands from among the new operands.
    unsigned j, f;
    for (j = 0, f = subOperands.size(); j < f; j++) {
      if (newOperands[i] == subOperands[j])
        break;
    }
    if (j < subOperands.size()) {
      newOperands[i] = (*sliceOps)[j];
    }
  }
  for (unsigned idx = 0, e = newOperands.size(); idx < e; idx++) {
    opInst->setOperand(idx, newOperands[idx]);
  }
}

/// Enum to set patterns of affine expr in tiled-layout map.
/// TileFloorDiv: <dim expr> div <tile size>
/// TileMod: <dim expr> mod <tile size>
/// TileNone: None of the above
/// Example:
/// #tiled_2d_128x256 = affine_map<(d0, d1)
///            -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>
/// "d0 div 128" and "d1 div 256" ==> TileFloorDiv
/// "d0 mod 128" and "d1 mod 256" ==> TileMod
enum TileExprPattern { TileFloorDiv, TileMod, TileNone };

/// Check if `map` is a tiled layout. In the tiled layout, specific k dimensions
/// being floordiv'ed by respective tile sizes appeare in a mod with the same
/// tile sizes, and no other expression involves those k dimensions. This
/// function stores a vector of tuples (`tileSizePos`) including AffineExpr for
/// tile size, positions of corresponding `floordiv` and `mod`. If it is not a
/// tiled layout, an empty vector is returned.
static LogicalResult getTileSizePos(
    AffineMap map,
    SmallVectorImpl<std::tuple<AffineExpr, unsigned, unsigned>> &tileSizePos) {
  // Create `floordivExprs` which is a vector of tuples including LHS and RHS of
  // `floordiv` and its position in `map` output.
  // Example: #tiled_2d_128x256 = affine_map<(d0, d1)
  //                -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>
  // In this example, `floordivExprs` includes {d0, 128, 0} and {d1, 256, 1}.
  SmallVector<std::tuple<AffineExpr, AffineExpr, unsigned>, 4> floordivExprs;
  unsigned pos = 0;
  for (AffineExpr expr : map.getResults()) {
    if (expr.getKind() == AffineExprKind::FloorDiv) {
      AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
      if (binaryExpr.getRHS().isa<AffineConstantExpr>())
        floordivExprs.emplace_back(
            std::make_tuple(binaryExpr.getLHS(), binaryExpr.getRHS(), pos));
    }
    pos++;
  }
  // Not tiled layout if `floordivExprs` is empty.
  if (floordivExprs.empty()) {
    tileSizePos = SmallVector<std::tuple<AffineExpr, unsigned, unsigned>>{};
    return success();
  }

  // Check if LHS of `floordiv` is used in LHS of `mod`. If not used, `map` is
  // not tiled layout.
  for (std::tuple<AffineExpr, AffineExpr, unsigned> fexpr : floordivExprs) {
    AffineExpr floordivExprLHS = std::get<0>(fexpr);
    AffineExpr floordivExprRHS = std::get<1>(fexpr);
    unsigned floordivPos = std::get<2>(fexpr);

    // Walk affinexpr of `map` output except `fexpr`, and check if LHS and RHS
    // of `fexpr` are used in LHS and RHS of `mod`. If LHS of `fexpr` is used
    // other expr, the map is not tiled layout. Example of non tiled layout:
    //   affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2 floordiv 256)>
    //   affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2 mod 128)>
    //   affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2 mod 256, d2 mod
    //   256)>
    bool found = false;
    pos = 0;
    for (AffineExpr expr : map.getResults()) {
      bool notTiled = false;
      if (pos != floordivPos) {
        expr.walk([&](AffineExpr e) {
          if (e == floordivExprLHS) {
            if (expr.getKind() == AffineExprKind::Mod) {
              AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
              // If LHS and RHS of `mod` are the same with those of floordiv.
              if (floordivExprLHS == binaryExpr.getLHS() &&
                  floordivExprRHS == binaryExpr.getRHS()) {
                // Save tile size (RHS of `mod`), and position of `floordiv` and
                // `mod` if same expr with `mod` is not found yet.
                if (!found) {
                  tileSizePos.emplace_back(
                      std::make_tuple(binaryExpr.getRHS(), floordivPos, pos));
                  found = true;
                } else {
                  // Non tiled layout: Have multilpe `mod` with the same LHS.
                  // eg. affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2
                  // mod 256, d2 mod 256)>
                  notTiled = true;
                }
              } else {
                // Non tiled layout: RHS of `mod` is different from `floordiv`.
                // eg. affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2
                // mod 128)>
                notTiled = true;
              }
            } else {
              // Non tiled layout: LHS is the same, but not `mod`.
              // eg. affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2
              // floordiv 256)>
              notTiled = true;
            }
          }
        });
      }
      if (notTiled) {
        tileSizePos = SmallVector<std::tuple<AffineExpr, unsigned, unsigned>>{};
        return success();
      }
      pos++;
    }
  }
  return success();
}

/// Check if `dim` dimension of memrefType with `layoutMap` becomes dynamic
/// after normalization. Dimensions that include dynamic dimensions in the map
/// output will become dynamic dimensions. Return true if `dim` is dynamic
/// dimension.
///
/// Example:
/// #map0 = affine_map<(d0, d1) -> (d0, d1 floordiv 32, d1 mod 32)>
///
/// If d1 is dynamic dimension, 2nd and 3rd dimension of map output are dynamic.
/// memref<4x?xf32, #map0>  ==>  memref<4x?x?xf32>
static bool
isNormalizedMemRefDynamicDim(unsigned dim, AffineMap layoutMap,
                             SmallVectorImpl<unsigned> &inMemrefTypeDynDims,
                             MLIRContext *context) {
  bool isDynamicDim = false;
  AffineExpr expr = layoutMap.getResults()[dim];
  // Check if affine expr of the dimension includes dynamic dimension of input
  // memrefType.
  expr.walk([&inMemrefTypeDynDims, &isDynamicDim, &context](AffineExpr e) {
    if (e.isa<AffineDimExpr>()) {
      for (unsigned dm : inMemrefTypeDynDims) {
        if (e == getAffineDimExpr(dm, context)) {
          isDynamicDim = true;
        }
      }
    }
  });
  return isDynamicDim;
}

/// Create affine expr to calculate dimension size for a tiled-layout map.
static AffineExpr createDimSizeExprForTiledLayout(AffineExpr oldMapOutput,
                                                  TileExprPattern pat) {
  // Create map output for the patterns.
  // "floordiv <tile size>" ==> "ceildiv <tile size>"
  // "mod <tile size>" ==> "<tile size>"
  AffineExpr newMapOutput;
  AffineBinaryOpExpr binaryExpr = nullptr;
  switch (pat) {
  case TileExprPattern::TileMod:
    binaryExpr = oldMapOutput.cast<AffineBinaryOpExpr>();
    newMapOutput = binaryExpr.getRHS();
    break;
  case TileExprPattern::TileFloorDiv:
    binaryExpr = oldMapOutput.cast<AffineBinaryOpExpr>();
    newMapOutput = getAffineBinaryOpExpr(
        AffineExprKind::CeilDiv, binaryExpr.getLHS(), binaryExpr.getRHS());
    break;
  default:
    newMapOutput = oldMapOutput;
  }
  return newMapOutput;
}

/// Create new maps to calculate each dimension size of `newMemRefType`, and
/// create `newDynamicSizes` from them by using AffineApplyOp.
///
/// Steps for normalizing dynamic memrefs for a tiled layout map
/// Example:
///    #map0 = affine_map<(d0, d1) -> (d0, d1 floordiv 32, d1 mod 32)>
///    %0 = dim %arg0, %c1 :memref<4x?xf32>
///    %1 = alloc(%0) : memref<4x?xf32, #map0>
///
/// (Before this function)
/// 1. Check if `map`(#map0) is a tiled layout using `getTileSizePos()`. Only
/// single layout map is supported.
///
/// 2. Create normalized memrefType using `isNormalizedMemRefDynamicDim()`. It
/// is memref<4x?x?xf32> in the above example.
///
/// (In this function)
/// 3. Create new maps to calculate each dimension of the normalized memrefType
/// using `createDimSizeExprForTiledLayout()`. In the tiled layout, the
/// dimension size can be calculated by replacing "floordiv <tile size>" with
/// "ceildiv <tile size>" and "mod <tile size>" with "<tile size>".
/// - New map in the above example
///   #map0 = affine_map<(d0, d1) -> (d0)>
///   #map1 = affine_map<(d0, d1) -> (d1 ceildiv 32)>
///   #map2 = affine_map<(d0, d1) -> (32)>
///
/// 4. Create AffineApplyOp to apply the new maps. The output of AffineApplyOp
/// is used in dynamicSizes of new AllocOp.
///   %0 = dim %arg0, %c1 : memref<4x?xf32>
///   %c4 = constant 4 : index
///   %1 = affine.apply #map1(%c4, %0)
///   %2 = affine.apply #map2(%c4, %0)
static void createNewDynamicSizes(MemRefType oldMemRefType,
                                  MemRefType newMemRefType, AffineMap map,
                                  memref::AllocOp *allocOp, OpBuilder b,
                                  SmallVectorImpl<Value> &newDynamicSizes) {
  // Create new input for AffineApplyOp.
  SmallVector<Value, 4> inAffineApply;
  ArrayRef<int64_t> oldMemRefShape = oldMemRefType.getShape();
  unsigned dynIdx = 0;
  for (unsigned d = 0; d < oldMemRefType.getRank(); ++d) {
    if (oldMemRefShape[d] < 0) {
      // Use dynamicSizes of allocOp for dynamic dimension.
      inAffineApply.emplace_back(allocOp->dynamicSizes()[dynIdx]);
      dynIdx++;
    } else {
      // Create ConstantOp for static dimension.
      Attribute constantAttr =
          b.getIntegerAttr(b.getIndexType(), oldMemRefShape[d]);
      inAffineApply.emplace_back(
          b.create<ConstantOp>(allocOp->getLoc(), constantAttr));
    }
  }

  // Create new map to calculate each dimension size of new memref for each
  // original map output. Only for dynamic dimesion of `newMemRefType`.
  unsigned newDimIdx = 0;
  ArrayRef<int64_t> newMemRefShape = newMemRefType.getShape();
  SmallVector<std::tuple<AffineExpr, unsigned, unsigned>> tileSizePos;
  (void)getTileSizePos(map, tileSizePos);
  for (AffineExpr expr : map.getResults()) {
    if (newMemRefShape[newDimIdx] < 0) {
      // Create new maps to calculate each dimension size of new memref.
      enum TileExprPattern pat = TileExprPattern::TileNone;
      for (auto pos : tileSizePos) {
        if (newDimIdx == std::get<1>(pos))
          pat = TileExprPattern::TileFloorDiv;
        else if (newDimIdx == std::get<2>(pos))
          pat = TileExprPattern::TileMod;
      }
      AffineExpr newMapOutput = createDimSizeExprForTiledLayout(expr, pat);
      AffineMap newMap =
          AffineMap::get(map.getNumInputs(), map.getNumSymbols(), newMapOutput);
      Value affineApp =
          b.create<AffineApplyOp>(allocOp->getLoc(), newMap, inAffineApply);
      newDynamicSizes.emplace_back(affineApp);
    }
    newDimIdx++;
  }
}

// TODO: Currently works for static memrefs with a single layout map.
LogicalResult mlir::normalizeMemRef(memref::AllocOp *allocOp) {
  MemRefType memrefType = allocOp->getType();
  OpBuilder b(*allocOp);

  // Fetch a new memref type after normalizing the old memref to have an
  // identity map layout.
  MemRefType newMemRefType =
      normalizeMemRefType(memrefType, b, allocOp->symbolOperands().size());
  if (newMemRefType == memrefType)
    // Either memrefType already had an identity map or the map couldn't be
    // transformed to an identity map.
    return failure();

  Value oldMemRef = allocOp->getResult();

  SmallVector<Value, 4> symbolOperands(allocOp->symbolOperands());
  AffineMap layoutMap = memrefType.getAffineMaps().front();
  memref::AllocOp newAlloc;
  // Check if `layoutMap` is a tiled layout. Only single layout map is
  // supported for normalizing dynamic memrefs.
  SmallVector<std::tuple<AffineExpr, unsigned, unsigned>> tileSizePos;
  (void)getTileSizePos(layoutMap, tileSizePos);
  if (newMemRefType.getNumDynamicDims() > 0 && !tileSizePos.empty()) {
    MemRefType oldMemRefType = oldMemRef.getType().cast<MemRefType>();
    SmallVector<Value, 4> newDynamicSizes;
    createNewDynamicSizes(oldMemRefType, newMemRefType, layoutMap, allocOp, b,
                          newDynamicSizes);
    // Add the new dynamic sizes in new AllocOp.
    newAlloc =
        b.create<memref::AllocOp>(allocOp->getLoc(), newMemRefType,
                                  newDynamicSizes, allocOp->alignmentAttr());
  } else {
    newAlloc = b.create<memref::AllocOp>(allocOp->getLoc(), newMemRefType,
                                         allocOp->alignmentAttr());
  }
  // Replace all uses of the old memref.
  if (failed(replaceAllMemRefUsesWith(oldMemRef, /*newMemRef=*/newAlloc,
                                      /*extraIndices=*/{},
                                      /*indexRemap=*/layoutMap,
                                      /*extraOperands=*/{},
                                      /*symbolOperands=*/symbolOperands,
                                      /*domInstFilter=*/nullptr,
                                      /*postDomInstFilter=*/nullptr,
                                      /*allowDereferencingOps=*/true))) {
    // If it failed (due to escapes for example), bail out.
    newAlloc.erase();
    return failure();
  }
  // Replace any uses of the original alloc op and erase it. All remaining uses
  // have to be dealloc's; RAMUW above would've failed otherwise.
  assert(llvm::all_of(oldMemRef.getUsers(), [](Operation *op) {
    return isa<memref::DeallocOp>(op);
  }));
  oldMemRef.replaceAllUsesWith(newAlloc);
  allocOp->erase();
  return success();
}

MemRefType mlir::normalizeMemRefType(MemRefType memrefType, OpBuilder b,
                                     unsigned numSymbolicOperands) {
  unsigned rank = memrefType.getRank();
  if (rank == 0)
    return memrefType;

  ArrayRef<AffineMap> layoutMaps = memrefType.getAffineMaps();
  if (layoutMaps.empty() ||
      layoutMaps.front() == b.getMultiDimIdentityMap(rank)) {
    // Either no maps is associated with this memref or this memref has
    // a trivial (identity) map.
    return memrefType;
  }

  // We don't do any checks for one-to-one'ness; we assume that it is
  // one-to-one.

  // Normalize only static memrefs and dynamic memrefs with a tiled-layout map
  // for now.
  // TODO: Normalize the other types of dynamic memrefs.
  SmallVector<std::tuple<AffineExpr, unsigned, unsigned>> tileSizePos;
  (void)getTileSizePos(layoutMaps.front(), tileSizePos);
  if (memrefType.getNumDynamicDims() > 0 && tileSizePos.empty())
    return memrefType;

  // We have a single map that is not an identity map. Create a new memref
  // with the right shape and an identity layout map.
  ArrayRef<int64_t> shape = memrefType.getShape();
  // FlatAffineConstraint may later on use symbolicOperands.
  FlatAffineConstraints fac(rank, numSymbolicOperands);
  SmallVector<unsigned, 4> memrefTypeDynDims;
  for (unsigned d = 0; d < rank; ++d) {
    // Use constraint system only in static dimensions.
    if (shape[d] > 0) {
      fac.addConstantLowerBound(d, 0);
      fac.addConstantUpperBound(d, shape[d] - 1);
    } else {
      memrefTypeDynDims.emplace_back(d);
    }
  }
  // We compose this map with the original index (logical) space to derive
  // the upper bounds for the new index space.
  AffineMap layoutMap = layoutMaps.front();
  unsigned newRank = layoutMap.getNumResults();
  if (failed(fac.composeMatchingMap(layoutMap)))
    return memrefType;
  // TODO: Handle semi-affine maps.
  // Project out the old data dimensions.
  fac.projectOut(newRank, fac.getNumIds() - newRank - fac.getNumLocalIds());
  SmallVector<int64_t, 4> newShape(newRank);
  for (unsigned d = 0; d < newRank; ++d) {
    // Check if each dimension of normalized memrefType is dynamic.
    bool isDynDim = isNormalizedMemRefDynamicDim(
        d, layoutMap, memrefTypeDynDims, b.getContext());
    if (isDynDim) {
      newShape[d] = -1;
    } else {
      // The lower bound for the shape is always zero.
      auto ubConst = fac.getConstantUpperBound(d);
      // For a static memref and an affine map with no symbols, this is
      // always bounded.
      assert(ubConst.hasValue() && "should always have an upper bound");
      if (ubConst.getValue() < 0)
        // This is due to an invalid map that maps to a negative space.
        return memrefType;
      // If dimension of new memrefType is dynamic, the value is -1.
      newShape[d] = ubConst.getValue() + 1;
    }
  }

  // Create the new memref type after trivializing the old layout map.
  MemRefType newMemRefType =
      MemRefType::Builder(memrefType)
          .setShape(newShape)
          .setAffineMaps(b.getMultiDimIdentityMap(newRank));

  return newMemRefType;
}
