//===- OpenMPOffloadPrivatizationPrepare.cpp - Prepare OMP privatization --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <iterator>
#include <utility>

//===----------------------------------------------------------------------===//
// A pass that prepares OpenMP code for translation of delayed privatization
// in the context of deferred target tasks. Deferred target tasks are created
// when the nowait clause is used on the target directive.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "omp-prepare-for-offload-privatization"

namespace mlir {
namespace omp {

#define GEN_PASS_DEF_PREPAREFOROMPOFFLOADPRIVATIZATIONPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"

} // namespace omp
} // namespace mlir

using namespace mlir;
namespace {

//===----------------------------------------------------------------------===//
// PrepareForOMPOffloadPrivatizationPass
//===----------------------------------------------------------------------===//

class PrepareForOMPOffloadPrivatizationPass
    : public omp::impl::PrepareForOMPOffloadPrivatizationPassBase<
          PrepareForOMPOffloadPrivatizationPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // In this pass, we make host-allocated privatized variables persist for
    // deferred target tasks by copying them to the heap. Once the target task
    // is done, this heap memory is freed. Since all of this happens on the host
    // we can skip device modules.
    auto offloadModuleInterface =
        dyn_cast<omp::OffloadModuleInterface>(mod.getOperation());
    if (offloadModuleInterface && offloadModuleInterface.getIsTargetDevice())
      return;

    getOperation()->walk([&](omp::TargetOp targetOp) {
      if (!hasPrivateVars(targetOp) || !isTargetTaskDeferred(targetOp))
        return;
      IRRewriter rewriter(&getContext());
      OperandRange privateVars = targetOp.getPrivateVars();
      SmallVector<mlir::Value> newPrivVars;
      Value fakeDependVar;
      omp::TaskOp cleanupTaskOp;

      newPrivVars.reserve(privateVars.size());
      std::optional<ArrayAttr> privateSyms = targetOp.getPrivateSyms();
      for (auto [privVarIdx, privVarSymPair] :
           llvm::enumerate(llvm::zip_equal(privateVars, *privateSyms))) {
        Value privVar = std::get<0>(privVarSymPair);
        Attribute privSym = std::get<1>(privVarSymPair);

        omp::PrivateClauseOp privatizer = findPrivatizer(targetOp, privSym);
        if (!privatizer.needsMap()) {
          newPrivVars.push_back(privVar);
          continue;
        }
        bool isFirstPrivate = privatizer.getDataSharingType() ==
                              omp::DataSharingClauseType::FirstPrivate;

        Value mappedValue = targetOp.getMappedValueForPrivateVar(privVarIdx);
        auto mapInfoOp = cast<omp::MapInfoOp>(mappedValue.getDefiningOp());

        if (mapInfoOp.getMapCaptureType() == omp::VariableCaptureKind::ByCopy) {
          newPrivVars.push_back(privVar);
          continue;
        }

        // For deferred target tasks (!$omp target nowait), we need to keep
        // a copy of the original, i.e. host variable being privatized so
        // that it is available when the target task is eventually executed.
        // We do this by first allocating as much heap memory as is needed by
        // the original variable. Then, we use the init and copy regions of the
        // privatizer, an instance of omp::PrivateClauseOp to set up the heap-
        // allocated copy.
        // After the target task is done, we need to use the dealloc region
        // of the privatizer to clean up everything. We also need to free
        // the heap memory we allocated. But due to the deferred nature
        // of the target task, we cannot simply deallocate right after the
        // omp.target operation else we may end up freeing memory before
        // its eventual use by the target task. So, we create a dummy
        // dependence between the target task and new omp.task. In the omp.task,
        // we do all the cleanup. So, we end up with the following structure
        //
        // omp.target map_entries(..) ... nowait depend(out:fakeDependVar) {
        //   ...
        //   omp.terminator
        // }
        // omp.task depend(in: fakeDependVar) {
        //   /*cleanup_code*/
        //   omp.terminator
        // }
        // fakeDependVar is the address of the first heap-allocated copy of the
        // host variable being privatized.

        bool needsCleanupTask = !privatizer.getDeallocRegion().empty();

        // Allocate heap memory that corresponds to the type of memory
        // pointed to by varPtr
        // For boxchars this won't be a pointer. But, MapsForPrivatizedSymbols
        // should have mapped the pointer to the boxchar so use that as varPtr.
        Value varPtr = mapInfoOp.getVarPtr();
        Type varType = mapInfoOp.getVarType();
        bool isPrivatizedByValue =
            !isa<LLVM::LLVMPointerType>(privVar.getType());

        assert(isa<LLVM::LLVMPointerType>(varPtr.getType()));
        Value heapMem =
            allocateHeapMem(targetOp, varPtr, varType, mod, rewriter);
        if (!heapMem)
          targetOp.emitError(
              "Unable to allocate heap memory when trying to move "
              "a private variable out of the stack and into the "
              "heap for use by a deferred target task");

        if (needsCleanupTask && !fakeDependVar)
          fakeDependVar = heapMem;

        // The types of private vars should match before and after the
        // transformation. In particular, if the type is a pointer,
        // simply record the newly allocated malloc location as the
        // new private variable. If, however, the type is not a pointer
        // then, we need to load the value from the newly allocated
        // location. We'll insert that load later after we have updated
        // the malloc'd location with the contents of the original
        // variable.
        if (!isPrivatizedByValue)
          newPrivVars.push_back(heapMem);

        // We now need to copy the original private variable into the newly
        // allocated location in the heap.
        // Find the earliest insertion point for the copy. This will be before
        // the first in the list of omp::MapInfoOp instances that use varPtr.
        // After the copy these omp::MapInfoOp instances will refer to heapMem
        // instead.
        Operation *varPtrDefiningOp = varPtr.getDefiningOp();
        DenseSet<Operation *> users;
        if (varPtrDefiningOp) {
          users.insert(varPtrDefiningOp->user_begin(),
                       varPtrDefiningOp->user_end());
        } else {
          auto blockArg = cast<BlockArgument>(varPtr);
          users.insert(blockArg.user_begin(), blockArg.user_end());
        }
        auto usesVarPtr = [&users](Operation *op) -> bool {
          return users.count(op);
        };

        SmallVector<Operation *> chainOfOps;
        chainOfOps.push_back(mapInfoOp);
        for (auto member : mapInfoOp.getMembers()) {
          omp::MapInfoOp memberMap =
              cast<omp::MapInfoOp>(member.getDefiningOp());
          if (usesVarPtr(memberMap))
            chainOfOps.push_back(memberMap);
          if (memberMap.getVarPtrPtr()) {
            Operation *defOp = memberMap.getVarPtrPtr().getDefiningOp();
            if (defOp && usesVarPtr(defOp))
              chainOfOps.push_back(defOp);
          }
        }

        DominanceInfo dom;
        llvm::sort(chainOfOps, [&](Operation *l, Operation *r) {
          if (l == r)
            return false;
          return dom.properlyDominates(l, r);
        });

        rewriter.setInsertionPoint(chainOfOps.front());

        Operation *firstOp = chainOfOps.front();
        Location loc = firstOp->getLoc();

        // Create a llvm.func for 'region' that is marked always_inline and call
        // it.
        auto createAlwaysInlineFuncAndCallIt =
            [&](Region &region, llvm::StringRef funcName,
                llvm::ArrayRef<Value> args, bool returnsValue) -> Value {
          assert(!region.empty() && "region cannot be empty");
          LLVM::LLVMFuncOp func = createFuncOpForRegion(
              loc, mod, region, funcName, rewriter, returnsValue);
          auto call = LLVM::CallOp::create(rewriter, loc, func, args);
          return call.getResult();
        };

        Value moldArg, newArg;
        if (isPrivatizedByValue) {
          moldArg = LLVM::LoadOp::create(rewriter, loc, varType, varPtr);
          newArg = LLVM::LoadOp::create(rewriter, loc, varType, heapMem);
        } else {
          moldArg = varPtr;
          newArg = heapMem;
        }

        Value initializedVal;
        if (!privatizer.getInitRegion().empty())
          initializedVal = createAlwaysInlineFuncAndCallIt(
              privatizer.getInitRegion(),
              llvm::formatv("{0}_{1}", privatizer.getSymName(), "init").str(),
              {moldArg, newArg}, /*returnsValue=*/true);
        else
          initializedVal = newArg;

        if (isFirstPrivate && !privatizer.getCopyRegion().empty())
          initializedVal = createAlwaysInlineFuncAndCallIt(
              privatizer.getCopyRegion(),
              llvm::formatv("{0}_{1}", privatizer.getSymName(), "copy").str(),
              {moldArg, initializedVal}, /*returnsValue=*/true);

        if (isPrivatizedByValue)
          (void)LLVM::StoreOp::create(rewriter, loc, initializedVal, heapMem);

        // clone origOp, replace all uses of varPtr with heapMem and
        // erase origOp.
        auto cloneModifyAndErase = [&](Operation *origOp) -> Operation * {
          Operation *clonedOp = rewriter.clone(*origOp);
          rewriter.replaceAllOpUsesWith(origOp, clonedOp);
          rewriter.modifyOpInPlace(clonedOp, [&]() {
            clonedOp->replaceUsesOfWith(varPtr, heapMem);
          });
          rewriter.eraseOp(origOp);
          return clonedOp;
        };

        // Now that we have set up the heap-allocated copy of the private
        // variable, rewrite all the uses of the original variable with
        // the heap-allocated variable.
        rewriter.setInsertionPoint(targetOp);
        mapInfoOp = cast<omp::MapInfoOp>(cloneModifyAndErase(mapInfoOp));
        rewriter.setInsertionPoint(mapInfoOp);

        // Fix any members that may use varPtr to now use heapMem
        for (auto member : mapInfoOp.getMembers()) {
          auto memberMapInfoOp = cast<omp::MapInfoOp>(member.getDefiningOp());
          if (!usesVarPtr(memberMapInfoOp))
            continue;
          memberMapInfoOp =
              cast<omp::MapInfoOp>(cloneModifyAndErase(memberMapInfoOp));
          rewriter.setInsertionPoint(memberMapInfoOp);

          if (memberMapInfoOp.getVarPtrPtr()) {
            Operation *varPtrPtrdefOp =
                memberMapInfoOp.getVarPtrPtr().getDefiningOp();
            rewriter.setInsertionPoint(cloneModifyAndErase(varPtrPtrdefOp));
          }
        }

        // If the type of the private variable is not a pointer,
        // which is typically the case with !fir.boxchar types, then
        // we need to ensure that the new private variable is also
        // not a pointer. Insert a load from heapMem right before
        // targetOp.
        if (isPrivatizedByValue) {
          rewriter.setInsertionPoint(targetOp);
          auto newPrivVar = LLVM::LoadOp::create(rewriter, mapInfoOp.getLoc(),
                                                 varType, heapMem);
          newPrivVars.push_back(newPrivVar);
        }

        // Deallocate
        if (needsCleanupTask) {
          if (!cleanupTaskOp) {
            assert(fakeDependVar &&
                   "Need a valid value to set up a dependency");
            rewriter.setInsertionPointAfter(targetOp);
            omp::TaskOperands taskOperands;
            auto inDepend = omp::ClauseTaskDependAttr::get(
                rewriter.getContext(), omp::ClauseTaskDepend::taskdependin);
            taskOperands.dependKinds.push_back(inDepend);
            taskOperands.dependVars.push_back(fakeDependVar);
            cleanupTaskOp = omp::TaskOp::create(rewriter, loc, taskOperands);
            Block *taskBlock = rewriter.createBlock(&cleanupTaskOp.getRegion());
            rewriter.setInsertionPointToEnd(taskBlock);
            omp::TerminatorOp::create(rewriter, cleanupTaskOp.getLoc());
          }
          rewriter.setInsertionPointToStart(
              &*cleanupTaskOp.getRegion().getBlocks().begin());
          (void)createAlwaysInlineFuncAndCallIt(
              privatizer.getDeallocRegion(),
              llvm::formatv("{0}_{1}", privatizer.getSymName(), "dealloc")
                  .str(),
              {initializedVal}, /*returnsValue=*/false);
          llvm::FailureOr<LLVM::LLVMFuncOp> freeFunc =
              LLVM::lookupOrCreateFreeFn(rewriter, mod);
          assert(llvm::succeeded(freeFunc) &&
                 "Could not find free in the module");
          (void)LLVM::CallOp::create(rewriter, loc, freeFunc.value(),
                                     ValueRange{heapMem});
        }
      }
      assert(newPrivVars.size() == privateVars.size() &&
             "The number of private variables must match before and after "
             "transformation");
      if (fakeDependVar) {
        omp::ClauseTaskDependAttr outDepend = omp::ClauseTaskDependAttr::get(
            rewriter.getContext(), omp::ClauseTaskDepend::taskdependout);
        SmallVector<Attribute> newDependKinds;
        if (!targetOp.getDependVars().empty()) {
          std::optional<ArrayAttr> dependKinds = targetOp.getDependKinds();
          assert(dependKinds && "bad depend clause in omp::TargetOp");
          llvm::copy(*dependKinds, std::back_inserter(newDependKinds));
        }
        newDependKinds.push_back(outDepend);
        ArrayAttr newDependKindsAttr =
            ArrayAttr::get(rewriter.getContext(), newDependKinds);
        targetOp.getDependVarsMutable().append(fakeDependVar);
        targetOp.setDependKindsAttr(newDependKindsAttr);
      }
      rewriter.setInsertionPoint(targetOp);
      targetOp.getPrivateVarsMutable().clear();
      targetOp.getPrivateVarsMutable().assign(newPrivVars);
    });
  }

private:
  bool hasPrivateVars(omp::TargetOp targetOp) const {
    return !targetOp.getPrivateVars().empty();
  }

  bool isTargetTaskDeferred(omp::TargetOp targetOp) const {
    return targetOp.getNowait();
  }

  template <typename OpTy>
  omp::PrivateClauseOp findPrivatizer(OpTy op, Attribute privSym) const {
    SymbolRefAttr privatizerName = llvm::cast<SymbolRefAttr>(privSym);
    omp::PrivateClauseOp privatizer =
        SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(
            op, privatizerName);
    return privatizer;
  }

  // Get the (compile-time constant) size of varType as per the
  // given DataLayout dl.
  std::int64_t getSizeInBytes(const DataLayout &dl, Type varType) const {
    llvm::TypeSize size = dl.getTypeSize(varType);
    unsigned short alignment = dl.getTypeABIAlignment(varType);
    return llvm::alignTo(size, alignment);
  }

  LLVM::LLVMFuncOp getMalloc(ModuleOp mod, IRRewriter &rewriter) const {
    llvm::FailureOr<LLVM::LLVMFuncOp> mallocCall =
        LLVM::lookupOrCreateMallocFn(rewriter, mod, rewriter.getI64Type());
    assert(llvm::succeeded(mallocCall) &&
           "Could not find malloc in the module");
    return mallocCall.value();
  }

  Value allocateHeapMem(omp::TargetOp targetOp, Value privVar, Type varType,
                        ModuleOp mod, IRRewriter &rewriter) const {
    OpBuilder::InsertionGuard guard(rewriter);
    Value varPtr = privVar;
    Operation *definingOp = varPtr.getDefiningOp();
    BlockArgument blockArg;
    if (!definingOp) {
      blockArg = mlir::dyn_cast<BlockArgument>(varPtr);
      rewriter.setInsertionPointToStart(blockArg.getParentBlock());
    } else {
      rewriter.setInsertionPoint(definingOp);
    }
    Location loc = definingOp ? definingOp->getLoc() : blockArg.getLoc();
    LLVM::LLVMFuncOp mallocFn = getMalloc(mod, rewriter);

    assert(mod.getDataLayoutSpec() &&
           "MLIR module with no datalayout spec not handled yet");

    const DataLayout &dl = DataLayout(mod);
    std::int64_t distance = getSizeInBytes(dl, varType);

    Value sizeBytes = LLVM::ConstantOp::create(
        rewriter, loc, mallocFn.getFunctionType().getParamType(0), distance);

    auto mallocCallOp =
        LLVM::CallOp::create(rewriter, loc, mallocFn, ValueRange{sizeBytes});
    return mallocCallOp.getResult();
  }

  // Create a function for srcRegion and attribute it to be always_inline.
  // The big assumption here is that srcRegion is one of init, copy or dealloc
  // regions of a omp::PrivateClauseop. Accordingly, the return type is assumed
  // to either be the same as the types of the two arguments of the region (for
  // init and copy regions) or void as would be the case for dealloc regions.
  LLVM::LLVMFuncOp createFuncOpForRegion(Location loc, ModuleOp mod,
                                         Region &srcRegion,
                                         llvm::StringRef funcName,
                                         IRRewriter &rewriter,
                                         bool returnsValue = false) {

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(mod.getBody(), mod.getBody()->end());
    Region clonedRegion;
    IRMapping mapper;
    srcRegion.cloneInto(&clonedRegion, mapper);

    SmallVector<Type> paramTypes;
    llvm::copy(srcRegion.getArgumentTypes(), std::back_inserter(paramTypes));
    Type resultType = returnsValue
                          ? srcRegion.getArgument(0).getType()
                          : LLVM::LLVMVoidType::get(rewriter.getContext());
    LLVM::LLVMFunctionType funcType =
        LLVM::LLVMFunctionType::get(resultType, paramTypes);

    LLVM::LLVMFuncOp func =
        LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcType);
    func.setAlwaysInline(true);
    rewriter.inlineRegionBefore(clonedRegion, func.getRegion(),
                                func.getRegion().end());
    for (auto &block : func.getRegion().getBlocks()) {
      if (isa<omp::YieldOp>(block.getTerminator())) {
        omp::YieldOp yieldOp = cast<omp::YieldOp>(block.getTerminator());
        rewriter.setInsertionPoint(yieldOp);
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(yieldOp, TypeRange(),
                                                    yieldOp.getOperands());
      }
    }
    return func;
  }
};
} // namespace
