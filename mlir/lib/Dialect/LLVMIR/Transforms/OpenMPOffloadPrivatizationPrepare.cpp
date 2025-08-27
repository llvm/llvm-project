//===- OpenMPOffloadPrivatizationPrepare.cpp - Prepare for OpenMP Offload
// Privatization ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/OpenMPOffloadPrivatizationPrepare.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>
#include <utility>

//===----------------------------------------------------------------------===//
// A pass that prepares OpenMP code for translation of delayed privatization
// in the context of deferred target tasks. Deferred target tasks are created
// when the nowait clause is used on the target directive.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "omp-prepare-for-offload-privatization"
#define PDBGS() (llvm::dbgs() << "[" << DEBUG_TYPE << "]: ")

namespace mlir {
namespace LLVM {

#define GEN_PASS_DEF_PREPAREFOROMPOFFLOADPRIVATIZATIONPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

} // namespace LLVM
} // namespace mlir

using namespace mlir;
namespace {

//===----------------------------------------------------------------------===//
// OMPTargetPrepareDelayedPrivatizationPattern
//===----------------------------------------------------------------------===//

class OMPTargetPrepareDelayedPrivatizationPattern
    : public OpRewritePattern<omp::TargetOp> {
public:
  using OpRewritePattern<omp::TargetOp>::OpRewritePattern;

  // Match omp::TargetOp that have the following characteristics.
  // 1. have private vars which refer to local (stack) memory
  // 2. the target op has the nowait clause
  // In this case, we allocate memory for the privatized variable on the heap
  // and copy the original variable into this new heap allocation. We fix up
  // any omp::MapInfoOp instances that may be mapping the private variable.
  mlir::LogicalResult
  matchAndRewrite(omp::TargetOp targetOp,
                  PatternRewriter &rewriter) const override {
    if (!hasPrivateVars(targetOp) || !isTargetTaskDeferred(targetOp))
      return rewriter.notifyMatchFailure(
          targetOp,
          "targetOp does not have privateVars or does not need a target task");

    ModuleOp mod = targetOp->getParentOfType<ModuleOp>();
    LLVM::LLVMFuncOp llvmFunc = targetOp->getParentOfType<LLVM::LLVMFuncOp>();
    OperandRange privateVars = targetOp.getPrivateVars();
    mlir::SmallVector<mlir::Value> newPrivVars;

    newPrivVars.reserve(privateVars.size());
    std::optional<ArrayAttr> privateSyms = targetOp.getPrivateSyms();
    for (auto [privVarIdx, privVarSymPair] :
         llvm::enumerate(llvm::zip_equal(privateVars, *privateSyms))) {
      auto privVar = std::get<0>(privVarSymPair);
      auto privSym = std::get<1>(privVarSymPair);

      omp::PrivateClauseOp privatizer = findPrivatizer(targetOp, privSym);
      if (!privatizer.needsMap()) {
        newPrivVars.push_back(privVar);
        continue;
      }
      bool isFirstPrivate = privatizer.getDataSharingType() ==
                            omp::DataSharingClauseType::FirstPrivate;

      mlir::Value mappedValue =
          targetOp.getMappedValueForPrivateVar(privVarIdx);
      Operation *mapInfoOperation = mappedValue.getDefiningOp();
      auto mapInfoOp = mlir::cast<omp::MapInfoOp>(mapInfoOperation);

      if (mapInfoOp.getMapCaptureType() == omp::VariableCaptureKind::ByCopy) {
        newPrivVars.push_back(privVar);
        continue;
      }

      // Allocate heap memory that corresponds to the type of memory
      // pointed to by varPtr
      // TODO: For boxchars this likely wont be a pointer.
      mlir::Value varPtr = privVar;
      mlir::Value heapMem = allocateHeapMem(targetOp, privVar, mod, rewriter);
      if (!heapMem)
        return failure();

      newPrivVars.push_back(heapMem);

      // Find the earliest insertion point for the copy. This will be before
      // the first in the list of omp::MapInfoOp instances that use varPtr.
      // After the copy these omp::MapInfoOp instances will refer to heapMem
      // instead.
      Operation *varPtrDefiningOp = varPtr.getDefiningOp();
      std::set<Operation *> users;
      users.insert(varPtrDefiningOp->user_begin(),
                   varPtrDefiningOp->user_end());

      auto usesVarPtr = [&users](Operation *op) -> bool {
        return users.count(op);
      };
      SmallVector<Operation *> chainOfOps;
      chainOfOps.push_back(mapInfoOperation);
      if (!mapInfoOp.getMembers().empty()) {
        for (auto member : mapInfoOp.getMembers()) {
          if (usesVarPtr(member.getDefiningOp()))
            chainOfOps.push_back(member.getDefiningOp());

          omp::MapInfoOp memberMap =
              mlir::cast<omp::MapInfoOp>(member.getDefiningOp());
          if (memberMap.getVarPtrPtr() &&
              usesVarPtr(memberMap.getVarPtrPtr().getDefiningOp()))
            chainOfOps.push_back(memberMap.getVarPtrPtr().getDefiningOp());
        }
      }
      DominanceInfo dom;
      llvm::sort(chainOfOps, [&](Operation *l, Operation *r) {
        return dom.dominates(l, r);
      });

      rewriter.setInsertionPoint(chainOfOps.front());
      // Copy the value of the local variable into the heap-allocated location.
      mlir::Location loc = chainOfOps.front()->getLoc();
      mlir::Type varType = getElemType(varPtr);
      auto loadVal = rewriter.create<LLVM::LoadOp>(loc, varType, varPtr);
      LLVM_ATTRIBUTE_UNUSED auto storeInst =
          rewriter.create<LLVM::StoreOp>(loc, loadVal.getResult(), heapMem);

      using ReplacementEntry = std::pair<Operation *, Operation *>;
      llvm::SmallVector<ReplacementEntry> replRecord;
      auto cloneAndMarkForDeletion = [&](Operation *origOp) -> Operation * {
        Operation *clonedOp = rewriter.clone(*origOp);
        rewriter.replaceAllOpUsesWith(origOp, clonedOp);
        replRecord.push_back(std::make_pair(origOp, clonedOp));
        return clonedOp;
      };

      rewriter.setInsertionPoint(targetOp);
      rewriter.setInsertionPoint(cloneAndMarkForDeletion(mapInfoOperation));

      // Fix any members that may use varPtr to now use heapMem
      if (!mapInfoOp.getMembers().empty()) {
        for (auto member : mapInfoOp.getMembers()) {
          Operation *memberOperation = member.getDefiningOp();
          if (!usesVarPtr(memberOperation))
            continue;
          rewriter.setInsertionPoint(cloneAndMarkForDeletion(memberOperation));

          auto memberMapInfoOp = mlir::cast<omp::MapInfoOp>(memberOperation);
          if (memberMapInfoOp.getVarPtrPtr()) {
            Operation *varPtrPtrdefOp =
                memberMapInfoOp.getVarPtrPtr().getDefiningOp();

            // In the case of firstprivate, we have to do the following
            // 1. Allocate heap memory for the underlying data.
            // 2. Copy the original underlying data to the new memory allocated
            // on the heap.
            // 3. Put this new (heap) address in the originating
            // struct/descriptor

            // Consider the following sequence of omp.map.info and omp.target
            // operations.
            // %0 = llvm.getelementptr %19[0, 0]
            // %1 = omp.map.info var_ptr(%19 : !llvm.ptr, i32) ...
            //                   var_ptr_ptr(%0 : !llvm.ptr)  bounds(..)
            // %2 = omp.map.info var_ptr(%19 : !llvm.ptr, !desc_type)>) ...
            //                   members(%1 : [0] : !llvm.ptr) -> !llvm.ptr
            // omp.target nowait map_entries(%2 -> %arg5, %1 -> %arg8 : ..)
            //                   private(@privatizer %19 -> %arg9 [map_idx=1] :
            //                   !llvm.ptr) {
            // We need to allocate memory on the heap for the underlying pointer
            // which is stored at the var_ptr_ptr operand of %1. Then we need to
            // copy this pointer to the new heap allocated memory location.
            // Then, we need to store the address of the new heap location in
            // the originating struct/descriptor. So, we generate the following
            // (pseudo) MLIR code (Using the same names of mlir::Value instances
            // in the example as in the code below)
            //
            // %dataMalloc = malloc(totalSize)
            // %loadDataPtr = load %0 : !llvm.ptr -> !llvm.ptr
            // memcpy(%dataMalloc, %loadDataPtr, totalSize)
            // %newVarPtrPtrOp = llvm.getelementptr %heapMem[0, 0]
            // llvm.store %dataMalloc, %newVarPtrPtrOp
            // %1.cloned = omp.map.info var_ptr(%heapMem : !llvm.ptr, i32) ...
            //                          var_ptr_ptr(%newVarPtrPtrOp : !llvm.ptr)
            // %2.cloned = omp.map.info var_ptr(%heapMem : !llvm.ptr,
            //                                             !desc_type)>) ...
            //                          members(%1.cloned : [0] : !llvm.ptr)
            //             -> !llvm.ptr
            // omp.target nowait map_entries(%2.cloned -> %arg5,
            //                               %1.cloned -> %arg8 : ..)
            //            private(@privatizer %heapMem -> .. [map_idx=1] : ..) {

            if (isFirstPrivate) {
              assert(!memberMapInfoOp.getBounds().empty() &&
                     "empty bounds on member map of firstprivate variable");
              mlir::Location loc = memberMapInfoOp.getLoc();
              mlir::Value totalSize =
                  getSizeInBytes(memberMapInfoOp, mod, rewriter);
              auto dataMalloc = allocateHeapMem(loc, totalSize, mod, rewriter);
              auto loadDataPtr = rewriter.create<LLVM::LoadOp>(
                  loc, memberMapInfoOp.getVarPtrPtr().getType(),
                  memberMapInfoOp.getVarPtrPtr());
              LLVM_ATTRIBUTE_UNUSED auto memcpy =
                  rewriter.create<mlir::LLVM::MemcpyOp>(
                      loc, dataMalloc.getResult(), loadDataPtr.getResult(),
                      totalSize, /*isVolatile=*/false);
              Operation *newVarPtrPtrOp = rewriter.clone(*varPtrPtrdefOp);
              rewriter.replaceAllUsesExcept(memberMapInfoOp.getVarPtrPtr(),
                                            newVarPtrPtrOp->getOpResult(0),
                                            loadDataPtr);
              rewriter.modifyOpInPlace(newVarPtrPtrOp, [&]() {
                newVarPtrPtrOp->replaceUsesOfWith(varPtr, heapMem);
              });
              LLVM_ATTRIBUTE_UNUSED auto storePtr =
                  rewriter.create<LLVM::StoreOp>(loc, dataMalloc.getResult(),
                                                 newVarPtrPtrOp->getResult(0));
            } else
              rewriter.setInsertionPoint(
                  cloneAndMarkForDeletion(varPtrPtrdefOp));
          }
        }
      }

      for (auto repl : replRecord) {
        Operation *origOp = repl.first;
        Operation *clonedOp = repl.second;
        rewriter.modifyOpInPlace(
            clonedOp, [&]() { clonedOp->replaceUsesOfWith(varPtr, heapMem); });
        rewriter.eraseOp(origOp);
      }
    }
    assert(newPrivVars.size() == privateVars.size() &&
           "The number of private variables must match before and after "
           "transformation");

    rewriter.setInsertionPoint(targetOp);
    Operation *newOp = rewriter.clone(*targetOp.getOperation());
    omp::TargetOp newTargetOp = mlir::cast<omp::TargetOp>(newOp);
    rewriter.modifyOpInPlace(newTargetOp, [&]() {
      newTargetOp.getPrivateVarsMutable().assign(newPrivVars);
    });
    rewriter.replaceOp(targetOp, newTargetOp);
    return mlir::success();
  }

private:
  bool hasPrivateVars(omp::TargetOp targetOp) const {
    return !targetOp.getPrivateVars().empty();
  }

  bool isTargetTaskDeferred(omp::TargetOp targetOp) const {
    return targetOp.getNowait();
  }

  template <typename OpTy>
  omp::PrivateClauseOp findPrivatizer(OpTy op, mlir::Attribute privSym) const {
    SymbolRefAttr privatizerName = llvm::cast<SymbolRefAttr>(privSym);
    omp::PrivateClauseOp privatizer =
        SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(
            op, privatizerName);
    return privatizer;
  }

  template <typename OpType>
  mlir::Type getElemType(OpType op) const {
    return op.getElemType();
  }

  mlir::Type getElemType(mlir::Value varPtr) const {
    Operation *definingOp = unwrapAddrSpaceCast(varPtr.getDefiningOp());
    assert((mlir::isa<LLVM::AllocaOp, LLVM::GEPOp>(definingOp)) &&
           "getElemType in PrepareForOMPOffloadPrivatizationPass can deal only "
           "with Alloca or GEP for now");
    if (auto allocaOp = mlir::dyn_cast<LLVM::AllocaOp>(definingOp))
      return getElemType(allocaOp);
    // TODO: get rid of this because GEPOp.getElemType() is not the right thing
    // to use.
    if (auto gepOp = mlir::dyn_cast<LLVM::GEPOp>(definingOp))
      return getElemType(gepOp);
    return mlir::Type{};
  }

  mlir::Operation *unwrapAddrSpaceCast(Operation *op) const {
    if (!mlir::isa<LLVM::AddrSpaceCastOp>(op))
      return op;
    mlir::LLVM::AddrSpaceCastOp addrSpaceCastOp =
        mlir::cast<LLVM::AddrSpaceCastOp>(op);
    return unwrapAddrSpaceCast(addrSpaceCastOp.getArg().getDefiningOp());
  }

  // Get the (compile-time constant) size of varType as per the
  // given DataLayout dl.
  std::int64_t getSizeInBytes(const mlir::DataLayout &dl,
                              mlir::Type varType) const {
    llvm::TypeSize size = dl.getTypeSize(varType);
    unsigned short alignment = dl.getTypeABIAlignment(varType);
    return llvm::alignTo(size, alignment);
  }

  // Generate code to get the size of data being mapped from the bounds
  // of mapInfoOp
  mlir::Value getSizeInBytes(omp::MapInfoOp mapInfoOp, ModuleOp mod,
                             PatternRewriter &rewriter) const {
    mlir::Location loc = mapInfoOp.getLoc();
    mlir::Type llvmInt64Ty = rewriter.getI64Type();
    mlir::Value constOne =
        rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Ty, 1);
    mlir::Value elementCount = constOne;
    // TODO: Consider using  boundsOp.getExtent() if available.
    for (auto bounds : mapInfoOp.getBounds()) {
      auto boundsOp = mlir::cast<omp::MapBoundsOp>(bounds.getDefiningOp());
      elementCount = rewriter.create<LLVM::MulOp>(
          loc, llvmInt64Ty, elementCount,
          rewriter.create<LLVM::AddOp>(
              loc, llvmInt64Ty,
              (rewriter.create<LLVM::SubOp>(loc, llvmInt64Ty,
                                            boundsOp.getUpperBound(),
                                            boundsOp.getLowerBound())),
              constOne));
    }
    const mlir::DataLayout &dl = mlir::DataLayout(mod);
    std::int64_t elemSize = getSizeInBytes(dl, mapInfoOp.getVarType());
    mlir::Value elemSizeV =
        rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Ty, elemSize);
    return rewriter.create<LLVM::MulOp>(loc, llvmInt64Ty, elementCount,
                                        elemSizeV);
  }

  LLVM::LLVMFuncOp getMalloc(ModuleOp mod, PatternRewriter &rewriter) const {
    llvm::FailureOr<mlir::LLVM::LLVMFuncOp> mallocCall =
        LLVM::lookupOrCreateMallocFn(rewriter, mod, rewriter.getI64Type());
    assert(llvm::succeeded(mallocCall) &&
           "Could not find malloc in the module");
    return mallocCall.value();
  }

  template <typename OpTy>
  mlir::Value allocateHeapMem(OpTy targetOp, mlir::Value privVar, ModuleOp mod,
                              PatternRewriter &rewriter) const {
    mlir::Value varPtr = privVar;
    Operation *definingOp = varPtr.getDefiningOp();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(definingOp);
    LLVM::LLVMFuncOp mallocFn = getMalloc(mod, rewriter);

    mlir::Location loc = definingOp->getLoc();
    mlir::Type varType = getElemType(varPtr);
    assert(mod.getDataLayoutSpec() &&
           "MLIR module with no datalayout spec not handled yet");
    const mlir::DataLayout &dl = mlir::DataLayout(mod);
    std::int64_t distance = getSizeInBytes(dl, varType);
    mlir::Value sizeBytes = rewriter.create<LLVM::ConstantOp>(
        loc, mallocFn.getFunctionType().getParamType(0), distance);

    auto mallocCallOp =
        rewriter.create<LLVM::CallOp>(loc, mallocFn, ValueRange{sizeBytes});
    return mallocCallOp.getResult();
  }

  LLVM::CallOp allocateHeapMem(mlir::Location loc, mlir::Value size,
                               ModuleOp mod, PatternRewriter &rewriter) const {
    LLVM::LLVMFuncOp mallocFn = getMalloc(mod, rewriter);
    return rewriter.create<LLVM::CallOp>(loc, mallocFn, ValueRange{size});
  }
};

//===----------------------------------------------------------------------===//
// PrepareForOMPOffloadPrivatizationPass
//===----------------------------------------------------------------------===//

struct PrepareForOMPOffloadPrivatizationPass
    : public LLVM::impl::PrepareForOMPOffloadPrivatizationPassBase<
          PrepareForOMPOffloadPrivatizationPass> {

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    MLIRContext &context = getContext();
    ModuleOp mod = func->getParentOfType<ModuleOp>();

    // FunctionFilteringPass removes bounds arguments from omp.map.info
    // operations. We require bounds else our pass asserts. But, that's only for
    // maps in functions that are on the host. So, skip functions being compiled
    // for the target.
    auto offloadModuleInterface =
        mlir::dyn_cast<omp::OffloadModuleInterface>(mod.getOperation());
    if (offloadModuleInterface && offloadModuleInterface.getIsTargetDevice()) {
      return;
    }

    RewritePatternSet patterns(&context);
    patterns.add<OMPTargetPrepareDelayedPrivatizationPattern>(&context);

    if (mlir::failed(
            applyPatternsGreedily(func, std::move(patterns),
                                  GreedyRewriteConfig().setStrictness(
                                      GreedyRewriteStrictness::ExistingOps)))) {
      emitError(func.getLoc(),
                "error in preparing targetOps for delayed privatization.");
      signalPassFailure();
    }
  }
};
} // namespace
