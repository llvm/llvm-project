//===- OpenMPOffloadPrivatizationPrepare.cpp - Prepare OMP privatization --===//
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
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <utility>

//===----------------------------------------------------------------------===//
// A pass that prepares OpenMP code for translation of delayed privatization
// in the context of deferred target tasks. Deferred target tasks are created
// when the nowait clause is used on the target directive.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "omp-prepare-for-offload-privatization"

namespace mlir {
namespace LLVM {

#define GEN_PASS_DEF_PREPAREFOROMPOFFLOADPRIVATIZATIONPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

} // namespace LLVM
} // namespace mlir

using namespace mlir;
namespace {

//===----------------------------------------------------------------------===//
// PrepareForOMPOffloadPrivatizationPass
//===----------------------------------------------------------------------===//

class PrepareForOMPOffloadPrivatizationPass
    : public LLVM::impl::PrepareForOMPOffloadPrivatizationPassBase<
          PrepareForOMPOffloadPrivatizationPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation()->getParentOfType<ModuleOp>();

    // FunctionFilteringPass removes bounds arguments from omp.map.info
    // operations. We require bounds else our pass asserts. But, that's only for
    // maps in functions that are on the host. So, skip functions being compiled
    // for the target.
    auto offloadModuleInterface =
        dyn_cast<omp::OffloadModuleInterface>(mod.getOperation());
    if (offloadModuleInterface && offloadModuleInterface.getIsTargetDevice())
      return;

    getOperation()->walk([&](omp::TargetOp targetOp) {
      if (!hasPrivateVars(targetOp) || !isTargetTaskDeferred(targetOp))
        return;
      IRRewriter rewriter(&getContext());
      ModuleOp mod = targetOp->getParentOfType<ModuleOp>();
      OperandRange privateVars = targetOp.getPrivateVars();
      SmallVector<mlir::Value> newPrivVars;

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

        Value mappedValue =
            targetOp.getMappedValueForPrivateVar(privVarIdx);
        Operation *mapInfoOperation = mappedValue.getDefiningOp();
        auto mapInfoOp = cast<omp::MapInfoOp>(mapInfoOperation);

        if (mapInfoOp.getMapCaptureType() == omp::VariableCaptureKind::ByCopy) {
          newPrivVars.push_back(privVar);
          continue;
        }

        // Allocate heap memory that corresponds to the type of memory
        // pointed to by varPtr
        // For boxchars this won't be a pointer. But, MapsForPrivatizedSymbols
        // should have mapped the pointer the boxchar so use that as varPtr.
        Value varPtr = privVar;
        if (!isa<LLVM::LLVMPointerType>(privVar.getType()))
          varPtr = mapInfoOp.getVarPtr();

        assert(isa<LLVM::LLVMPointerType>(varPtr.getType()));
        Value heapMem = allocateHeapMem(targetOp, varPtr, mod, rewriter);
        if (!heapMem)
          targetOp.emitError(
              "Unable to allocate heap memory when trying to move "
              "a private variable out of the stack and into the "
              "heap for use by a deferred target task");

        // The types of private vars should match before and after the
        // transformation. In particular, if the type is a pointer,
        // simply record the newly allocated malloc location as the
        // new private variable. If, however, the type is not a pointer
        // then, we need to load the value from the newly allocated
        // location. We'll inser that load later after we have updated
        // the malloc'd location with the contents of the original
        // variable.
        if (isa<LLVM::LLVMPointerType>(privVar.getType()))
          newPrivVars.push_back(heapMem);

        // Find the earliest insertion point for the copy. This will be before
        // the first in the list of omp::MapInfoOp instances that use varPtr.
        // After the copy these omp::MapInfoOp instances will refer to heapMem
        // instead.
        Operation *varPtrDefiningOp = varPtr.getDefiningOp();
        DenseSet<Operation *> users;
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
                cast<omp::MapInfoOp>(member.getDefiningOp());
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
        // Copy the value of the local variable into the heap-allocated
        // location.
        Operation *firstOp = chainOfOps.front();
        Location loc = firstOp->getLoc();
        Type varType = getElemType(varPtr);


        // // auto loadVal = rewriter.create<LLVM::LoadOp>(loc, varType, varPtr);
        // // (void)rewriter.create<LLVM::StoreOp>(loc, loadVal.getResult(), heapMem);
        #if 0
        Region &initRegion = privatizer.getInitRegion();
        assert(!initRegion.empty() && "initRegion cannot be empty");
        Block &entryBlock = initRegion.front();
        Block *insertBlock = firstOp->getBlock();
        Block *newBlock = insertBlock->splitBlock(firstOp);
        Region *destRegion = firstOp->getParentRegion();
        IRMapping irMap;
        irMap.map(varPtr, entryBlock.getArgument(0));
        irMap.map(heapMem, entryBlock.getArgument(1));

        LDBG() << "Operation being walked before cloning the init region\n\n";
        LLVM_DEBUG(llvm::dbgs() << getOperation() << "\n");
        initRegion.cloneInto(destRegion, Region::iterator(newBlock), irMap);
        LDBG() << "Operation being walked after cloning the init region\n";
        LLVM_DEBUG(llvm::dbgs() << getOperation() << "\n");
        //        rewriter.setInsertionPointToEnd(insertBlock);
        // LLVM::BrOp::create(rewriter, loc,
        //            , );
#else
        // Todo: Handle boxchar (by value)
        Region &initRegion = privatizer.getInitRegion();
        assert(!initRegion.empty() && "initRegion cannot be empty");
        LLVM::LLVMFuncOp initFunc = createFuncOpForRegion(
            loc, mod, initRegion,
            llvm::formatv("{0}_{1}", privatizer.getSymName(), "init").str(),
            firstOp, rewriter);

        rewriter.create<LLVM::CallOp>(loc, initFunc, ValueRange{varPtr, heapMem});
#endif
        using ReplacementEntry = std::pair<Operation *, Operation *>;
        llvm::SmallVector<ReplacementEntry> replRecord;
        auto cloneAndMarkForDeletion = [&](Operation *origOp) -> Operation * {
          Operation *clonedOp = rewriter.clone(*origOp);
          rewriter.replaceAllOpUsesWith(origOp, clonedOp);
          replRecord.push_back(std::make_pair(origOp, clonedOp));
          return clonedOp;
        };

        if (isFirstPrivate) {
          Region &copyRegion = privatizer.getCopyRegion();
          assert(!copyRegion.empty() && "copyRegion cannot be empty");
          LLVM::LLVMFuncOp copyFunc = createFuncOpForRegion(
              loc, mod,  copyRegion,
              llvm::formatv("{0}_{1}", privatizer.getSymName(), "copy").str(),
              firstOp, rewriter);
          rewriter.create<LLVM::CallOp>(loc, copyFunc, ValueRange{varPtr, heapMem});
        }

        rewriter.setInsertionPoint(targetOp);
        rewriter.setInsertionPoint(cloneAndMarkForDeletion(mapInfoOperation));

        // Fix any members that may use varPtr to now use heapMem
        if (!mapInfoOp.getMembers().empty()) {
          for (auto member : mapInfoOp.getMembers()) {
            Operation *memberOperation = member.getDefiningOp();
            if (!usesVarPtr(memberOperation))
              continue;
            rewriter.setInsertionPoint(
                cloneAndMarkForDeletion(memberOperation));

            auto memberMapInfoOp = cast<omp::MapInfoOp>(memberOperation);
            if (memberMapInfoOp.getVarPtrPtr()) {
              Operation *varPtrPtrdefOp =
                  memberMapInfoOp.getVarPtrPtr().getDefiningOp();
              rewriter.setInsertionPoint(
                  cloneAndMarkForDeletion(varPtrPtrdefOp));
            }
          }
        }

        for (auto repl : replRecord) {
          Operation *origOp = repl.first;
          Operation *clonedOp = repl.second;
          rewriter.modifyOpInPlace(clonedOp, [&]() {
            clonedOp->replaceUsesOfWith(varPtr, heapMem);
          });
          rewriter.eraseOp(origOp);
        }

        // If the type of the private variable is not a pointer,
        // which is typically the case with !fir.boxchar types, then
        // we need to ensure that the new private variable is also
        // not a pointer. Insert a load from heapMem right before
        // targetOp.
        if (!isa<LLVM::LLVMPointerType>(privVar.getType())) {
          rewriter.setInsertionPoint(targetOp);
          auto newPrivVar = rewriter.create<LLVM::LoadOp>(mapInfoOp.getLoc(),
                                                          varType, heapMem);
          newPrivVars.push_back(newPrivVar);
        }
      }
      assert(newPrivVars.size() == privateVars.size() &&
             "The number of private variables must match before and after "
             "transformation");

      rewriter.setInsertionPoint(targetOp);
      Operation *newOp = rewriter.clone(*targetOp.getOperation());
      omp::TargetOp newTargetOp = cast<omp::TargetOp>(newOp);
      rewriter.modifyOpInPlace(newTargetOp, [&]() {
        newTargetOp.getPrivateVarsMutable().assign(newPrivVars);
      });
      rewriter.replaceOp(targetOp, newTargetOp);
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

  template <typename OpType>
  Type getElemType(OpType op) const {
    return op.getElemType();
  }

  Type getElemType(Value varPtr) const {
    Operation *definingOp = unwrapAddrSpaceCast(varPtr.getDefiningOp());
    assert((isa<LLVM::AllocaOp, LLVM::GEPOp>(definingOp)) &&
           "getElemType in PrepareForOMPOffloadPrivatizationPass can deal only "
           "with Alloca or GEP for now");
    if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(definingOp))
      return getElemType(allocaOp);
    // TODO: get rid of this because GEPOp.getElemType() is not the right thing
    // to use.
    if (auto gepOp = dyn_cast<LLVM::GEPOp>(definingOp))
      return getElemType(gepOp);
    return Type{};
  }

  Operation *unwrapAddrSpaceCast(Operation *op) const {
    if (!isa<LLVM::AddrSpaceCastOp>(op))
      return op;
    LLVM::AddrSpaceCastOp addrSpaceCastOp =
        cast<LLVM::AddrSpaceCastOp>(op);
    return unwrapAddrSpaceCast(addrSpaceCastOp.getArg().getDefiningOp());
  }

  // Get the (compile-time constant) size of varType as per the
  // given DataLayout dl.
  std::int64_t getSizeInBytes(const DataLayout &dl,
                              Type varType) const {
    llvm::TypeSize size = dl.getTypeSize(varType);
    unsigned short alignment = dl.getTypeABIAlignment(varType);
    return llvm::alignTo(size, alignment);
  }

  // Generate code to get the size of data being mapped from the bounds
  // of mapInfoOp
  Value getSizeInBytes(omp::MapInfoOp mapInfoOp, ModuleOp mod,
                             IRRewriter &rewriter) const {
    Location loc = mapInfoOp.getLoc();
    Type llvmInt64Ty = rewriter.getI64Type();
    Value constOne =
        rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Ty, 1);
    Value elementCount = constOne;
    // TODO: Consider using  boundsOp.getExtent() if available.
    for (auto bounds : mapInfoOp.getBounds()) {
      auto boundsOp = cast<omp::MapBoundsOp>(bounds.getDefiningOp());
      elementCount = rewriter.create<LLVM::MulOp>(
          loc, llvmInt64Ty, elementCount,
          rewriter.create<LLVM::AddOp>(
              loc, llvmInt64Ty,
              (rewriter.create<LLVM::SubOp>(loc, llvmInt64Ty,
                                            boundsOp.getUpperBound(),
                                            boundsOp.getLowerBound())),
              constOne));
    }
    const DataLayout &dl = DataLayout(mod);
    std::int64_t elemSize = getSizeInBytes(dl, mapInfoOp.getVarType());
    Value elemSizeV =
        rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Ty, elemSize);
    return rewriter.create<LLVM::MulOp>(loc, llvmInt64Ty, elementCount,
                                        elemSizeV);
  }

  LLVM::LLVMFuncOp getMalloc(ModuleOp mod, IRRewriter &rewriter) const {
    llvm::FailureOr<LLVM::LLVMFuncOp> mallocCall =
        LLVM::lookupOrCreateMallocFn(rewriter, mod, rewriter.getI64Type());
    assert(llvm::succeeded(mallocCall) &&
           "Could not find malloc in the module");
    return mallocCall.value();
  }

  template <typename OpTy>
  Value allocateHeapMem(OpTy targetOp, Value privVar, ModuleOp mod,
                        IRRewriter &rewriter) const {
    Value varPtr = privVar;
    Operation *definingOp = varPtr.getDefiningOp();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(definingOp);
    LLVM::LLVMFuncOp mallocFn = getMalloc(mod, rewriter);

    Location loc = definingOp->getLoc();
    Type varType = getElemType(varPtr);
    assert(mod.getDataLayoutSpec() &&
           "MLIR module with no datalayout spec not handled yet");
    const DataLayout &dl = DataLayout(mod);
    std::int64_t distance = getSizeInBytes(dl, varType);
    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(
        loc, mallocFn.getFunctionType().getParamType(0), distance);

    auto mallocCallOp =
        rewriter.create<LLVM::CallOp>(loc, mallocFn, ValueRange{sizeBytes});
    return mallocCallOp.getResult();
  }

  LLVM::CallOp allocateHeapMem(Location loc, Value size,
                               ModuleOp mod, IRRewriter &rewriter) const {
    LLVM::LLVMFuncOp mallocFn = getMalloc(mod, rewriter);
    return rewriter.create<LLVM::CallOp>(loc, mallocFn, ValueRange{size});
  }
  LLVM::LLVMFuncOp createFuncOpForRegion(Location loc, ModuleOp mod,
                                         Region &srcRegion,
                                         llvm::StringRef funcName,
                                         Operation *insertPt,
                                         IRRewriter &rewriter) {

    OpBuilder::InsertionGuard guard(rewriter);
    MLIRContext *ctx = mod.getContext();
    rewriter.setInsertionPoint(mod.getBody(), mod.getBody()->end());
    Region clonedRegion;
    IRMapping mapper;
    srcRegion.cloneInto(&clonedRegion, mapper);
    SmallVector<Type> paramTypes = {srcRegion.getArgument(0).getType(),
                                    srcRegion.getArgument(1).getType()};
    LDBG() << "paramTypes are \n"
           << srcRegion.getArgument(0).getType() << "\n"
           << srcRegion.getArgument(1).getType() << "\n";
    LLVM::LLVMFunctionType funcType =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), paramTypes);

    LDBG() << "funcType is " << funcType << "\n";
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
                                                    Value());
      }
    }
    LDBG() << funcName << " is \n" << func << "\n";
    LLVM_DEBUG(llvm::dbgs() << "Module is \n" << mod << "\n");
    return func;
  }
};
} // namespace
