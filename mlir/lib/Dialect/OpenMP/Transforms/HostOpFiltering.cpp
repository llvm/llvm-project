//===- HostOpFiltering.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to swap stack allocations on the target
// device with device shared memory where applicable.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
namespace omp {
#define GEN_PASS_DEF_HOSTOPFILTERINGPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"
} // namespace omp
} // namespace mlir

using namespace mlir;

/// Some host operations, like \c llvm.mlir.addressof and constants, must remain
/// in the device module because they impact how device code is generated when
/// attached to an \c omp.target operation.
///
/// This function identifies the operations that need this special handling.
/// This includes cast-style operations to avoid losing information about the
/// original source of an operand.
static bool keepHostOpInDevice(Operation &op) {
  return isPure(&op) &&
         op.getDialect() ==
             op.getContext()->getLoadedDialect<LLVM::LLVMDialect>();
}

/// Add an \c omp.map.info operation and all its members recursively to the
/// output set to be later rewritten.
///
/// Dependencies across \c omp.map.info are maintained by ensuring dependencies
/// are added to the output sets before operations based on them.
static void collectRewrite(omp::MapInfoOp mapOp,
                           llvm::SetVector<omp::MapInfoOp> &rewrites) {
  for (Value member : mapOp.getMembers())
    collectRewrite(cast<omp::MapInfoOp>(member.getDefiningOp()), rewrites);

  rewrites.insert(mapOp);
}

/// Add the given value to a sorted set if it should be replaced by a
/// placeholder when used as an operand that must remain for the device.
///
/// Values that are block arguments of function operations are skipped, since
/// they will still be available after all rewrites are completed, and operands
/// of operations that need to remain on the host are recursively collected.
static void collectRewrite(Value value, llvm::SetVector<Value> &rewrites) {
  if ((isa<BlockArgument>(value) &&
       isa<FunctionOpInterface>(
           cast<BlockArgument>(value).getOwner()->getParentOp())) ||
      rewrites.contains(value))
    return;

  Operation *op = value.getDefiningOp();
  if (op && keepHostOpInDevice(*op))
    for (Value operand : op->getOperands())
      collectRewrite(operand, rewrites);

  rewrites.insert(value);
}

/// Provide the \c device_type of an \c omp.declare_target attribute, if
/// defined.
static std::optional<omp::DeclareTargetDeviceType>
getDeclareTargetDevice(Operation &op) {
  auto declareTargetOp = dyn_cast<omp::DeclareTargetInterface>(op);
  if (declareTargetOp && declareTargetOp.isDeclareTarget())
    return declareTargetOp.getDeclareTargetDeviceType();
  return std::nullopt;
}

namespace {
class HostOpFilteringPass
    : public omp::impl::HostOpFilteringPassBase<HostOpFilteringPass> {
public:
  HostOpFilteringPass() = default;

  void runOnOperation() override {
    auto op = dyn_cast<omp::OffloadModuleInterface>(getOperation());
    if (!op || !op.getIsTargetDevice())
      return;

    op->walk<WalkOrder::PreOrder>([&](LLVM::LLVMFuncOp funcOp) {
      omp::DeclareTargetDeviceType declareType =
          getDeclareTargetDevice(*funcOp.getOperation())
              .value_or(omp::DeclareTargetDeviceType::host);

      // Only process host function definitions.
      if (funcOp.isExternal() ||
          declareType != omp::DeclareTargetDeviceType::host)
        return WalkResult::advance();

      if (failed(rewriteHostFunction(funcOp))) {
        funcOp.emitOpError() << "could not filter host-only operations";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // Make non-declare target globals internal for the device. They cannot be
    // deleted, because they are needed in order to properly lower map clauses.
    // However, no uses will remain in the device module, so we make them
    // internal to prevent link time redefinitions.
    op->walk([&](LLVM::GlobalOp globalOp) {
      if (!getDeclareTargetDevice(*globalOp.getOperation()).has_value())
        globalOp.setLinkage(LLVM::Linkage::Internal);
    });
  }

private:
  /// Rewrite the given host device function containing \c omp.target
  /// operations, to remove host-only operations that are not used by device
  /// codegen.
  ///
  /// It is based on the expected form of an MLIR module lowered to where it can
  /// be directly translated to LLVM IR and it performs the following mutations:
  ///   - Removes all returned values from the function.
  ///   - \c omp.target operations are moved to the end of the function. If they
  ///     are nested inside of any other operations, they are hoisted out of
  ///     them.
  ///   - \c depend, \c device, \c dyn_groupprivate, \c if and \c in_reduction
  ///     clauses are removed from these target functions. Values used to
  ///     initialize other clauses are replaced by placeholders as follows:
  ///     - Values defined by block arguments are replaced by placeholders only
  ///       if they are not attached to the parent function. In that case, they
  ///       are passed unmodified.
  ///     - Pure operations of the LLVM dialect are maintained, and any value
  ///       operands they might have are also replaced by placeholders following
  ///       the same rules.
  ///     - Other values are replaced by new function arguments.
  ///   - \c omp.map.info operations associated to these target regions are
  ///     preserved. These are moved above all \c omp.target and sorted to
  ///     satisfy dependencies among them.
  ///   - \c bounds arguments are removed from \c omp.map.info operations.
  ///   - \c var_ptr and \c var_ptr_ptr arguments of \c omp.map.info are
  ///     replaced by placeholders as described above.
  ///   - Every other operation not located inside of an \c omp.target is
  ///     removed.
  LogicalResult rewriteHostFunction(LLVM::LLVMFuncOp funcOp) {
    Region &region = funcOp.getFunctionBody();
    LLVM::LLVMFunctionType functionType = funcOp.getFunctionType();

    // Collect target operations inside of the function.
    llvm::SmallVector<omp::TargetOp> targetOps;
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Skip the inside of omp.target regions, since these contain device code.
      if (auto targetOp = dyn_cast<omp::TargetOp>(op)) {
        targetOps.push_back(targetOp);
        return WalkResult::skip();
      }

      // Replace omp.target_data entry block argument uses with the value used
      // to initialize the associated omp.map.info operation. This way,
      // references are still valid once the omp.target operation has been
      // extracted out of the omp.target_data region.
      if (auto targetDataOp = dyn_cast<omp::TargetDataOp>(op)) {
        llvm::SmallVector<std::pair<Value, BlockArgument>> argPairs;
        cast<omp::BlockArgOpenMPOpInterface>(*targetDataOp)
            .getBlockArgsPairs(argPairs);
        for (auto [operand, blockArg] : argPairs) {
          auto mapInfo = cast<omp::MapInfoOp>(operand.getDefiningOp());
          blockArg.replaceAllUsesWith(mapInfo.getVarPtr());
        }
      }
      return WalkResult::advance();
    });

    // Make a temporary clone of the parent function with an empty region,
    // and update all references to entry block arguments to those of the new
    // region. Users of these arguments will later either be moved to the new
    // region or deleted when the original region is replaced by the new.
    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(funcOp);
    Operation *newFuncOp = builder.cloneWithoutRegions(funcOp);
    Block &block = newFuncOp->getRegion(0).emplaceBlock();

    llvm::SmallVector<Location> locs;
    locs.reserve(region.getNumArguments());
    llvm::transform(region.getArguments(), std::back_inserter(locs),
                    [](const BlockArgument &arg) { return arg.getLoc(); });
    block.addArguments(region.getArgumentTypes(), locs);

    for (auto [oldArg, newArg] :
         llvm::zip_equal(region.getArguments(), block.getArguments()))
      oldArg.replaceAllUsesWith(newArg);

    // Collect omp.map.info ops while satisfying interdependencies and remove
    // operands that aren't used by target device codegen.
    //
    // This logic must be updated whenever operands to omp.target change.
    llvm::SetVector<Value> rewriteValues;
    llvm::SetVector<omp::MapInfoOp> mapInfos;
    for (omp::TargetOp targetOp : targetOps) {
      assert(targetOp.getHostEvalVars().empty() &&
             "unexpected host_eval in target device module");

      // Variables unused by the device.
      targetOp.getDependVarsMutable().clear();
      targetOp.setDependKindsAttr(nullptr);
      targetOp.getDependIteratedMutable().clear();
      targetOp.setDependIteratedKindsAttr(nullptr);
      targetOp.getDeviceMutable().clear();
      targetOp.getDynGroupprivateSizeMutable().clear();
      targetOp.getIfExprMutable().clear();
      targetOp.getInReductionVarsMutable().clear();
      targetOp.setInReductionByrefAttr(nullptr);
      targetOp.setInReductionSymsAttr(nullptr);

      // TODO: Clear some of these operands rather than rewriting them,
      // depending on whether they are needed by device codegen once support for
      // them is fully implemented.
      for (Value allocVar : targetOp.getAllocateVars())
        collectRewrite(allocVar, rewriteValues);
      for (Value allocVar : targetOp.getAllocatorVars())
        collectRewrite(allocVar, rewriteValues);
      for (Value isDevPtr : targetOp.getIsDevicePtrVars())
        collectRewrite(isDevPtr, rewriteValues);
      for (Value mapVar : targetOp.getHasDeviceAddrVars())
        collectRewrite(cast<omp::MapInfoOp>(mapVar.getDefiningOp()), mapInfos);
      for (Value mapVar : targetOp.getMapVars())
        collectRewrite(cast<omp::MapInfoOp>(mapVar.getDefiningOp()), mapInfos);
      for (Value privateVar : targetOp.getPrivateVars())
        collectRewrite(privateVar, rewriteValues);
      for (Value threadLimit : targetOp.getThreadLimitVars())
        collectRewrite(threadLimit, rewriteValues);
    }

    // Move omp.map.info ops to the new block and collect dependencies.
    for (omp::MapInfoOp mapOp : mapInfos) {
      collectRewrite(mapOp.getVarPtr(), rewriteValues);

      if (Value varPtrPtr = mapOp.getVarPtrPtr())
        collectRewrite(varPtrPtr, rewriteValues);

      // Bounds are not used during target device codegen.
      mapOp.getBoundsMutable().clear();
      mapOp->moveBefore(&block, block.end());
    }

    builder.setInsertionPointToStart(&block);

    // We don't actually need the proper initialization for all operands, but
    // rather just to maintain the basic form of omp.target operations. We
    // create new function arguments as placeholders for rewritten values.
    llvm::SmallVector<Type> newFnArgTypes(functionType.getParams());
    for (Value value : rewriteValues) {
      Value rewriteValue;
      Operation *definingOp = value.getDefiningOp();
      if (definingOp && keepHostOpInDevice(*definingOp)) {
        rewriteValue = builder.clone(*value.getDefiningOp())->getResult(0);
      } else {
        rewriteValue = block.addArgument(value.getType(), value.getLoc());
        newFnArgTypes.push_back(rewriteValue.getType());
      }
      value.replaceAllUsesWith(rewriteValue);
    }

    // Move target operations to the end of the new block.
    for (omp::TargetOp targetOp : targetOps)
      targetOp->moveBefore(&block, block.end());

    // Add terminator to the new block.
    builder.setInsertionPointToEnd(&block);
    LLVM::ReturnOp::create(builder, funcOp.getLoc(), ValueRange());

    // Replace old region with the new one, now only containing the required
    // operations, and remove the temporary operation clone.
    region.takeBody(newFuncOp->getRegion(0));
    newFuncOp->erase();

    // Update function type after modifying the terminator and argument list.
    funcOp.setType(LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(&getContext()), newFnArgTypes));

    return success();
  }
};
} // namespace
