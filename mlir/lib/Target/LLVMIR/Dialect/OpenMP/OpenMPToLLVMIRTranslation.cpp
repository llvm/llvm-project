//===- OpenMPToLLVMIRTranslation.cpp - Translate OpenMP dialect to LLVM IR-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR OpenMP dialect and LLVM
// IR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMPCommon.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <any>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <utility>

using namespace mlir;

namespace {
static llvm::omp::ScheduleKind
convertToScheduleKind(std::optional<omp::ClauseScheduleKind> schedKind) {
  if (!schedKind.has_value())
    return llvm::omp::OMP_SCHEDULE_Default;
  switch (schedKind.value()) {
  case omp::ClauseScheduleKind::Static:
    return llvm::omp::OMP_SCHEDULE_Static;
  case omp::ClauseScheduleKind::Dynamic:
    return llvm::omp::OMP_SCHEDULE_Dynamic;
  case omp::ClauseScheduleKind::Guided:
    return llvm::omp::OMP_SCHEDULE_Guided;
  case omp::ClauseScheduleKind::Auto:
    return llvm::omp::OMP_SCHEDULE_Auto;
  case omp::ClauseScheduleKind::Runtime:
    return llvm::omp::OMP_SCHEDULE_Runtime;
  }
  llvm_unreachable("unhandled schedule clause argument");
}

/// ModuleTranslation stack frame for OpenMP operations. This keeps track of the
/// insertion points for allocas.
class OpenMPAllocaStackFrame
    : public LLVM::ModuleTranslation::StackFrameBase<OpenMPAllocaStackFrame> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpenMPAllocaStackFrame)

  explicit OpenMPAllocaStackFrame(llvm::OpenMPIRBuilder::InsertPointTy allocaIP)
      : allocaInsertPoint(allocaIP) {}
  llvm::OpenMPIRBuilder::InsertPointTy allocaInsertPoint;
};

/// ModuleTranslation stack frame containing the partial mapping between MLIR
/// values and their LLVM IR equivalents.
class OpenMPVarMappingStackFrame
    : public LLVM::ModuleTranslation::StackFrameBase<
          OpenMPVarMappingStackFrame> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpenMPVarMappingStackFrame)

  explicit OpenMPVarMappingStackFrame(
      const DenseMap<Value, llvm::Value *> &mapping)
      : mapping(mapping) {}

  DenseMap<Value, llvm::Value *> mapping;
};
} // namespace

/// Find the insertion point for allocas given the current insertion point for
/// normal operations in the builder.
static llvm::OpenMPIRBuilder::InsertPointTy
findAllocaInsertPoint(llvm::IRBuilderBase &builder,
                      const LLVM::ModuleTranslation &moduleTranslation) {
  // If there is an alloca insertion point on stack, i.e. we are in a nested
  // operation and a specific point was provided by some surrounding operation,
  // use it.
  llvm::OpenMPIRBuilder::InsertPointTy allocaInsertPoint;
  WalkResult walkResult = moduleTranslation.stackWalk<OpenMPAllocaStackFrame>(
      [&](const OpenMPAllocaStackFrame &frame) {
        allocaInsertPoint = frame.allocaInsertPoint;
        return WalkResult::interrupt();
      });
  if (walkResult.wasInterrupted())
    return allocaInsertPoint;

  // Otherwise, insert to the entry block of the surrounding function.
  // If the current IRBuilder InsertPoint is the function's entry, it cannot
  // also be used for alloca insertion which would result in insertion order
  // confusion. Create a new BasicBlock for the Builder and use the entry block
  // for the allocs.
  // TODO: Create a dedicated alloca BasicBlock at function creation such that
  // we do not need to move the current InertPoint here.
  if (builder.GetInsertBlock() ==
      &builder.GetInsertBlock()->getParent()->getEntryBlock()) {
    assert(builder.GetInsertPoint() == builder.GetInsertBlock()->end() &&
           "Assuming end of basic block");
    llvm::BasicBlock *entryBB = llvm::BasicBlock::Create(
        builder.getContext(), "entry", builder.GetInsertBlock()->getParent(),
        builder.GetInsertBlock()->getNextNode());
    builder.CreateBr(entryBB);
    builder.SetInsertPoint(entryBB);
  }

  llvm::BasicBlock &funcEntryBlock =
      builder.GetInsertBlock()->getParent()->getEntryBlock();
  return llvm::OpenMPIRBuilder::InsertPointTy(
      &funcEntryBlock, funcEntryBlock.getFirstInsertionPt());
}

/// Converts the given region that appears within an OpenMP dialect operation to
/// LLVM IR, creating a branch from the `sourceBlock` to the entry block of the
/// region, and a branch from any block with an successor-less OpenMP terminator
/// to `continuationBlock`. Populates `continuationBlockPHIs` with the PHI nodes
/// of the continuation block if provided.
static llvm::BasicBlock *convertOmpOpRegions(
    Region &region, StringRef blockName, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation, LogicalResult &bodyGenStatus,
    SmallVectorImpl<llvm::PHINode *> *continuationBlockPHIs = nullptr) {
  llvm::BasicBlock *continuationBlock =
      splitBB(builder, true, "omp.region.cont");
  llvm::BasicBlock *sourceBlock = builder.GetInsertBlock();

  llvm::LLVMContext &llvmContext = builder.getContext();
  for (Block &bb : region) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        llvmContext, blockName, builder.GetInsertBlock()->getParent(),
        builder.GetInsertBlock()->getNextNode());
    moduleTranslation.mapBlock(&bb, llvmBB);
  }

  llvm::Instruction *sourceTerminator = sourceBlock->getTerminator();

  // Terminators (namely YieldOp) may be forwarding values to the region that
  // need to be available in the continuation block. Collect the types of these
  // operands in preparation of creating PHI nodes.
  SmallVector<llvm::Type *> continuationBlockPHITypes;
  bool operandsProcessed = false;
  unsigned numYields = 0;
  for (Block &bb : region.getBlocks()) {
    if (omp::YieldOp yield = dyn_cast<omp::YieldOp>(bb.getTerminator())) {
      if (!operandsProcessed) {
        for (unsigned i = 0, e = yield->getNumOperands(); i < e; ++i) {
          continuationBlockPHITypes.push_back(
              moduleTranslation.convertType(yield->getOperand(i).getType()));
        }
        operandsProcessed = true;
      } else {
        assert(continuationBlockPHITypes.size() == yield->getNumOperands() &&
               "mismatching number of values yielded from the region");
        for (unsigned i = 0, e = yield->getNumOperands(); i < e; ++i) {
          llvm::Type *operandType =
              moduleTranslation.convertType(yield->getOperand(i).getType());
          (void)operandType;
          assert(continuationBlockPHITypes[i] == operandType &&
                 "values of mismatching types yielded from the region");
        }
      }
      numYields++;
    }
  }

  // Insert PHI nodes in the continuation block for any values forwarded by the
  // terminators in this region.
  if (!continuationBlockPHITypes.empty())
    assert(
        continuationBlockPHIs &&
        "expected continuation block PHIs if converted regions yield values");
  if (continuationBlockPHIs) {
    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    continuationBlockPHIs->reserve(continuationBlockPHITypes.size());
    builder.SetInsertPoint(continuationBlock, continuationBlock->begin());
    for (llvm::Type *ty : continuationBlockPHITypes)
      continuationBlockPHIs->push_back(builder.CreatePHI(ty, numYields));
  }

  // Convert blocks one by one in topological order to ensure
  // defs are converted before uses.
  SetVector<Block *> blocks = getBlocksSortedByDominance(region);
  for (Block *bb : blocks) {
    llvm::BasicBlock *llvmBB = moduleTranslation.lookupBlock(bb);
    // Retarget the branch of the entry block to the entry block of the
    // converted region (regions are single-entry).
    if (bb->isEntryBlock()) {
      assert(sourceTerminator->getNumSuccessors() == 1 &&
             "provided entry block has multiple successors");
      assert(sourceTerminator->getSuccessor(0) == continuationBlock &&
             "ContinuationBlock is not the successor of the entry block");
      sourceTerminator->setSuccessor(0, llvmBB);
    }

    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    if (failed(
            moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder))) {
      bodyGenStatus = failure();
      return continuationBlock;
    }

    // Special handling for `omp.yield` and `omp.terminator` (we may have more
    // than one): they return the control to the parent OpenMP dialect operation
    // so replace them with the branch to the continuation block. We handle this
    // here to avoid relying inter-function communication through the
    // ModuleTranslation class to set up the correct insertion point. This is
    // also consistent with MLIR's idiom of handling special region terminators
    // in the same code that handles the region-owning operation.
    Operation *terminator = bb->getTerminator();
    if (isa<omp::TerminatorOp, omp::YieldOp>(terminator)) {
      builder.CreateBr(continuationBlock);

      for (unsigned i = 0, e = terminator->getNumOperands(); i < e; ++i)
        (*continuationBlockPHIs)[i]->addIncoming(
            moduleTranslation.lookupValue(terminator->getOperand(i)), llvmBB);
    }
  }
  // After all blocks have been traversed and values mapped, connect the PHI
  // nodes to the results of preceding blocks.
  LLVM::detail::connectPHINodes(region, moduleTranslation);

  // Remove the blocks and values defined in this region from the mapping since
  // they are not visible outside of this region. This allows the same region to
  // be converted several times, that is cloned, without clashes, and slightly
  // speeds up the lookups.
  moduleTranslation.forgetMapping(region);

  return continuationBlock;
}

/// Convert ProcBindKind from MLIR-generated enum to LLVM enum.
static llvm::omp::ProcBindKind getProcBindKind(omp::ClauseProcBindKind kind) {
  switch (kind) {
  case omp::ClauseProcBindKind::Close:
    return llvm::omp::ProcBindKind::OMP_PROC_BIND_close;
  case omp::ClauseProcBindKind::Master:
    return llvm::omp::ProcBindKind::OMP_PROC_BIND_master;
  case omp::ClauseProcBindKind::Primary:
    return llvm::omp::ProcBindKind::OMP_PROC_BIND_primary;
  case omp::ClauseProcBindKind::Spread:
    return llvm::omp::ProcBindKind::OMP_PROC_BIND_spread;
  }
  llvm_unreachable("Unknown ClauseProcBindKind kind");
}

/// Converts an OpenMP 'masked' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpMasked(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  auto maskedOp = cast<omp::MaskedOp>(opInst);
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // MaskedOp has only one region associated with it.
    auto &region = maskedOp.getRegion();
    builder.restoreIP(codeGenIP);
    convertOmpOpRegions(region, "omp.masked.region", builder, moduleTranslation,
                        bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::Value *filterVal = nullptr;
  if (auto filterVar = maskedOp.getFilteredThreadId()) {
    filterVal = moduleTranslation.lookupValue(filterVar);
  } else {
    llvm::LLVMContext &llvmContext = builder.getContext();
    filterVal =
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext), /*V=*/0);
  }
  assert(filterVal != nullptr);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createMasked(
      ompLoc, bodyGenCB, finiCB, filterVal));
  return success();
}

/// Converts an OpenMP 'master' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpMaster(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // MasterOp has only one region associated with it.
    auto &region = cast<omp::MasterOp>(opInst).getRegion();
    builder.restoreIP(codeGenIP);
    convertOmpOpRegions(region, "omp.master.region", builder, moduleTranslation,
                        bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createMaster(
      ompLoc, bodyGenCB, finiCB));
  return success();
}

/// Converts an OpenMP 'critical' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpCritical(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto criticalOp = cast<omp::CriticalOp>(opInst);
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // CriticalOp has only one region associated with it.
    auto &region = cast<omp::CriticalOp>(opInst).getRegion();
    builder.restoreIP(codeGenIP);
    convertOmpOpRegions(region, "omp.critical.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
  llvm::Constant *hint = nullptr;

  // If it has a name, it probably has a hint too.
  if (criticalOp.getNameAttr()) {
    // The verifiers in OpenMP Dialect guarentee that all the pointers are
    // non-null
    auto symbolRef = cast<SymbolRefAttr>(criticalOp.getNameAttr());
    auto criticalDeclareOp =
        SymbolTable::lookupNearestSymbolFrom<omp::CriticalDeclareOp>(criticalOp,
                                                                     symbolRef);
    hint =
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext),
                               static_cast<int>(criticalDeclareOp.getHint()));
  }
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createCritical(
      ompLoc, bodyGenCB, finiCB, criticalOp.getName().value_or(""), hint));
  return success();
}

/// Populates `reductions` with reduction declarations used in the given loop.
template <typename T>
static void
collectReductionDecls(T loop,
                      SmallVectorImpl<omp::DeclareReductionOp> &reductions) {
  std::optional<ArrayAttr> attr = loop.getReductionSyms();
  if (!attr)
    return;

  reductions.reserve(reductions.size() + loop.getNumReductionVars());
  for (auto symbolRef : attr->getAsRange<SymbolRefAttr>()) {
    reductions.push_back(
        SymbolTable::lookupNearestSymbolFrom<omp::DeclareReductionOp>(
            loop, symbolRef));
  }
}

/// Translates the blocks contained in the given region and appends them to at
/// the current insertion point of `builder`. The operations of the entry block
/// are appended to the current insertion block. If set, `continuationBlockArgs`
/// is populated with translated values that correspond to the values
/// omp.yield'ed from the region.
static LogicalResult inlineConvertOmpRegions(
    Region &region, StringRef blockName, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation,
    SmallVectorImpl<llvm::Value *> *continuationBlockArgs = nullptr) {
  if (region.empty())
    return success();

  // Special case for single-block regions that don't create additional blocks:
  // insert operations without creating additional blocks.
  if (llvm::hasSingleElement(region)) {
    llvm::Instruction *potentialTerminator =
        builder.GetInsertBlock()->empty() ? nullptr
                                          : &builder.GetInsertBlock()->back();

    if (potentialTerminator && potentialTerminator->isTerminator())
      potentialTerminator->removeFromParent();
    moduleTranslation.mapBlock(&region.front(), builder.GetInsertBlock());

    if (failed(moduleTranslation.convertBlock(
            region.front(), /*ignoreArguments=*/true, builder)))
      return failure();

    // The continuation arguments are simply the translated terminator operands.
    if (continuationBlockArgs)
      llvm::append_range(
          *continuationBlockArgs,
          moduleTranslation.lookupValues(region.front().back().getOperands()));

    // Drop the mapping that is no longer necessary so that the same region can
    // be processed multiple times.
    moduleTranslation.forgetMapping(region);

    if (potentialTerminator && potentialTerminator->isTerminator()) {
      llvm::BasicBlock *block = builder.GetInsertBlock();
      if (block->empty()) {
        // this can happen for really simple reduction init regions e.g.
        // %0 = llvm.mlir.constant(0 : i32) : i32
        // omp.yield(%0 : i32)
        // because the llvm.mlir.constant (MLIR op) isn't converted into any
        // llvm op
        potentialTerminator->insertInto(block, block->begin());
      } else {
        potentialTerminator->insertAfter(&block->back());
      }
    }

    return success();
  }

  LogicalResult bodyGenStatus = success();
  SmallVector<llvm::PHINode *> phis;
  llvm::BasicBlock *continuationBlock = convertOmpOpRegions(
      region, blockName, builder, moduleTranslation, bodyGenStatus, &phis);
  if (failed(bodyGenStatus))
    return failure();
  if (continuationBlockArgs)
    llvm::append_range(*continuationBlockArgs, phis);
  builder.SetInsertPoint(continuationBlock,
                         continuationBlock->getFirstInsertionPt());
  return success();
}

namespace {
/// Owning equivalents of OpenMPIRBuilder::(Atomic)ReductionGen that are used to
/// store lambdas with capture.
using OwningReductionGen = std::function<llvm::OpenMPIRBuilder::InsertPointTy(
    llvm::OpenMPIRBuilder::InsertPointTy, llvm::Value *, llvm::Value *,
    llvm::Value *&)>;
using OwningAtomicReductionGen =
    std::function<llvm::OpenMPIRBuilder::InsertPointTy(
        llvm::OpenMPIRBuilder::InsertPointTy, llvm::Type *, llvm::Value *,
        llvm::Value *)>;
} // namespace

/// Create an OpenMPIRBuilder-compatible reduction generator for the given
/// reduction declaration. The generator uses `builder` but ignores its
/// insertion point.
static OwningReductionGen
makeReductionGen(omp::DeclareReductionOp decl, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  // The lambda is mutable because we need access to non-const methods of decl
  // (which aren't actually mutating it), and we must capture decl by-value to
  // avoid the dangling reference after the parent function returns.
  OwningReductionGen gen =
      [&, decl](llvm::OpenMPIRBuilder::InsertPointTy insertPoint,
                llvm::Value *lhs, llvm::Value *rhs,
                llvm::Value *&result) mutable {
        Region &reductionRegion = decl.getReductionRegion();
        moduleTranslation.mapValue(reductionRegion.front().getArgument(0), lhs);
        moduleTranslation.mapValue(reductionRegion.front().getArgument(1), rhs);
        builder.restoreIP(insertPoint);
        SmallVector<llvm::Value *> phis;
        if (failed(inlineConvertOmpRegions(reductionRegion,
                                           "omp.reduction.nonatomic.body",
                                           builder, moduleTranslation, &phis)))
          return llvm::OpenMPIRBuilder::InsertPointTy();
        assert(phis.size() == 1);
        result = phis[0];
        return builder.saveIP();
      };
  return gen;
}

/// Create an OpenMPIRBuilder-compatible atomic reduction generator for the
/// given reduction declaration. The generator uses `builder` but ignores its
/// insertion point. Returns null if there is no atomic region available in the
/// reduction declaration.
static OwningAtomicReductionGen
makeAtomicReductionGen(omp::DeclareReductionOp decl,
                       llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  if (decl.getAtomicReductionRegion().empty())
    return OwningAtomicReductionGen();

  // The lambda is mutable because we need access to non-const methods of decl
  // (which aren't actually mutating it), and we must capture decl by-value to
  // avoid the dangling reference after the parent function returns.
  OwningAtomicReductionGen atomicGen =
      [&, decl](llvm::OpenMPIRBuilder::InsertPointTy insertPoint, llvm::Type *,
                llvm::Value *lhs, llvm::Value *rhs) mutable {
        Region &atomicRegion = decl.getAtomicReductionRegion();
        moduleTranslation.mapValue(atomicRegion.front().getArgument(0), lhs);
        moduleTranslation.mapValue(atomicRegion.front().getArgument(1), rhs);
        builder.restoreIP(insertPoint);
        SmallVector<llvm::Value *> phis;
        if (failed(inlineConvertOmpRegions(atomicRegion,
                                           "omp.reduction.atomic.body", builder,
                                           moduleTranslation, &phis)))
          return llvm::OpenMPIRBuilder::InsertPointTy();
        assert(phis.empty());
        return builder.saveIP();
      };
  return atomicGen;
}

/// Converts an OpenMP 'ordered' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpOrdered(Operation &opInst, llvm::IRBuilderBase &builder,
                  LLVM::ModuleTranslation &moduleTranslation) {
  auto orderedOp = cast<omp::OrderedOp>(opInst);

  omp::ClauseDepend dependType = *orderedOp.getDoacrossDependType();
  bool isDependSource = dependType == omp::ClauseDepend::dependsource;
  unsigned numLoops = *orderedOp.getDoacrossNumLoops();
  SmallVector<llvm::Value *> vecValues =
      moduleTranslation.lookupValues(orderedOp.getDoacrossDependVars());

  size_t indexVecValues = 0;
  while (indexVecValues < vecValues.size()) {
    SmallVector<llvm::Value *> storeValues;
    storeValues.reserve(numLoops);
    for (unsigned i = 0; i < numLoops; i++) {
      storeValues.push_back(vecValues[indexVecValues]);
      indexVecValues++;
    }
    llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
        findAllocaInsertPoint(builder, moduleTranslation);
    llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
    builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createOrderedDepend(
        ompLoc, allocaIP, numLoops, storeValues, ".cnt.addr", isDependSource));
  }
  return success();
}

/// Converts an OpenMP 'ordered_region' operation into LLVM IR using
/// OpenMPIRBuilder.
static LogicalResult
convertOmpOrderedRegion(Operation &opInst, llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto orderedRegionOp = cast<omp::OrderedRegionOp>(opInst);

  // TODO: The code generation for ordered simd directive is not supported yet.
  if (orderedRegionOp.getParLevelSimd())
    return failure();

  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // OrderedOp has only one region associated with it.
    auto &region = cast<omp::OrderedRegionOp>(opInst).getRegion();
    builder.restoreIP(codeGenIP);
    convertOmpOpRegions(region, "omp.ordered.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(
      moduleTranslation.getOpenMPBuilder()->createOrderedThreadsSimd(
          ompLoc, bodyGenCB, finiCB, !orderedRegionOp.getParLevelSimd()));
  return bodyGenStatus;
}

namespace {
/// Contains the arguments for an LLVM store operation
struct DeferredStore {
  DeferredStore(llvm::Value *value, llvm::Value *address)
      : value(value), address(address) {}

  llvm::Value *value;
  llvm::Value *address;
};
} // namespace

/// Allocate space for privatized reduction variables.
/// `deferredStores` contains information to create store operations which needs
/// to be inserted after all allocas
template <typename T>
static LogicalResult
allocReductionVars(T loop, ArrayRef<BlockArgument> reductionArgs,
                   llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation,
                   llvm::OpenMPIRBuilder::InsertPointTy &allocaIP,
                   SmallVectorImpl<omp::DeclareReductionOp> &reductionDecls,
                   SmallVectorImpl<llvm::Value *> &privateReductionVariables,
                   DenseMap<Value, llvm::Value *> &reductionVariableMap,
                   SmallVectorImpl<DeferredStore> &deferredStores,
                   llvm::ArrayRef<bool> isByRefs) {
  llvm::IRBuilderBase::InsertPointGuard guard(builder);
  builder.SetInsertPoint(allocaIP.getBlock()->getTerminator());

  // delay creating stores until after all allocas
  deferredStores.reserve(loop.getNumReductionVars());

  for (std::size_t i = 0; i < loop.getNumReductionVars(); ++i) {
    Region &allocRegion = reductionDecls[i].getAllocRegion();
    if (isByRefs[i]) {
      if (allocRegion.empty())
        continue;

      SmallVector<llvm::Value *, 1> phis;
      if (failed(inlineConvertOmpRegions(allocRegion, "omp.reduction.alloc",
                                         builder, moduleTranslation, &phis)))
        return failure();
      assert(phis.size() == 1 && "expected one allocation to be yielded");

      builder.SetInsertPoint(allocaIP.getBlock()->getTerminator());

      // Allocate reduction variable (which is a pointer to the real reduction
      // variable allocated in the inlined region)
      llvm::Value *var = builder.CreateAlloca(
          moduleTranslation.convertType(reductionDecls[i].getType()));
      deferredStores.emplace_back(phis[0], var);

      privateReductionVariables[i] = var;
      moduleTranslation.mapValue(reductionArgs[i], phis[0]);
      reductionVariableMap.try_emplace(loop.getReductionVars()[i], phis[0]);
    } else {
      assert(allocRegion.empty() &&
             "allocaction is implicit for by-val reduction");
      llvm::Value *var = builder.CreateAlloca(
          moduleTranslation.convertType(reductionDecls[i].getType()));
      moduleTranslation.mapValue(reductionArgs[i], var);
      privateReductionVariables[i] = var;
      reductionVariableMap.try_emplace(loop.getReductionVars()[i], var);
    }
  }

  return success();
}

/// Map input arguments to reduction initialization region
template <typename T>
static void
mapInitializationArgs(T loop, LLVM::ModuleTranslation &moduleTranslation,
                      SmallVectorImpl<omp::DeclareReductionOp> &reductionDecls,
                      DenseMap<Value, llvm::Value *> &reductionVariableMap,
                      unsigned i) {
  // map input argument to the initialization region
  mlir::omp::DeclareReductionOp &reduction = reductionDecls[i];
  Region &initializerRegion = reduction.getInitializerRegion();
  Block &entry = initializerRegion.front();

  mlir::Value mlirSource = loop.getReductionVars()[i];
  llvm::Value *llvmSource = moduleTranslation.lookupValue(mlirSource);
  assert(llvmSource && "lookup reduction var");
  moduleTranslation.mapValue(reduction.getInitializerMoldArg(), llvmSource);

  if (entry.getNumArguments() > 1) {
    llvm::Value *allocation =
        reductionVariableMap.lookup(loop.getReductionVars()[i]);
    moduleTranslation.mapValue(reduction.getInitializerAllocArg(), allocation);
  }
}

/// Collect reduction info
template <typename T>
static void collectReductionInfo(
    T loop, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation,
    SmallVectorImpl<omp::DeclareReductionOp> &reductionDecls,
    SmallVectorImpl<OwningReductionGen> &owningReductionGens,
    SmallVectorImpl<OwningAtomicReductionGen> &owningAtomicReductionGens,
    const ArrayRef<llvm::Value *> privateReductionVariables,
    SmallVectorImpl<llvm::OpenMPIRBuilder::ReductionInfo> &reductionInfos) {
  unsigned numReductions = loop.getNumReductionVars();

  for (unsigned i = 0; i < numReductions; ++i) {
    owningReductionGens.push_back(
        makeReductionGen(reductionDecls[i], builder, moduleTranslation));
    owningAtomicReductionGens.push_back(
        makeAtomicReductionGen(reductionDecls[i], builder, moduleTranslation));
  }

  // Collect the reduction information.
  reductionInfos.reserve(numReductions);
  for (unsigned i = 0; i < numReductions; ++i) {
    llvm::OpenMPIRBuilder::ReductionGenAtomicCBTy atomicGen = nullptr;
    if (owningAtomicReductionGens[i])
      atomicGen = owningAtomicReductionGens[i];
    llvm::Value *variable =
        moduleTranslation.lookupValue(loop.getReductionVars()[i]);
    reductionInfos.push_back(
        {moduleTranslation.convertType(reductionDecls[i].getType()), variable,
         privateReductionVariables[i],
         /*EvaluationKind=*/llvm::OpenMPIRBuilder::EvalKind::Scalar,
         owningReductionGens[i],
         /*ReductionGenClang=*/nullptr, atomicGen});
  }
}

/// handling of DeclareReductionOp's cleanup region
static LogicalResult
inlineOmpRegionCleanup(llvm::SmallVectorImpl<Region *> &cleanupRegions,
                       llvm::ArrayRef<llvm::Value *> privateVariables,
                       LLVM::ModuleTranslation &moduleTranslation,
                       llvm::IRBuilderBase &builder, StringRef regionName,
                       bool shouldLoadCleanupRegionArg = true) {
  for (auto [i, cleanupRegion] : llvm::enumerate(cleanupRegions)) {
    if (cleanupRegion->empty())
      continue;

    // map the argument to the cleanup region
    Block &entry = cleanupRegion->front();

    llvm::Instruction *potentialTerminator =
        builder.GetInsertBlock()->empty() ? nullptr
                                          : &builder.GetInsertBlock()->back();
    if (potentialTerminator && potentialTerminator->isTerminator())
      builder.SetInsertPoint(potentialTerminator);
    llvm::Value *privateVarValue =
        shouldLoadCleanupRegionArg
            ? builder.CreateLoad(
                  moduleTranslation.convertType(entry.getArgument(0).getType()),
                  privateVariables[i])
            : privateVariables[i];

    moduleTranslation.mapValue(entry.getArgument(0), privateVarValue);

    if (failed(inlineConvertOmpRegions(*cleanupRegion, regionName, builder,
                                       moduleTranslation)))
      return failure();

    // clear block argument mapping in case it needs to be re-created with a
    // different source for another use of the same reduction decl
    moduleTranslation.forgetMapping(*cleanupRegion);
  }
  return success();
}

// TODO: not used by ParallelOp
template <class OP>
static LogicalResult createReductionsAndCleanup(
    OP op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation,
    llvm::OpenMPIRBuilder::InsertPointTy &allocaIP,
    SmallVectorImpl<omp::DeclareReductionOp> &reductionDecls,
    ArrayRef<llvm::Value *> privateReductionVariables, ArrayRef<bool> isByRef) {
  // Process the reductions if required.
  if (op.getNumReductionVars() == 0)
    return success();

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // Create the reduction generators. We need to own them here because
  // ReductionInfo only accepts references to the generators.
  SmallVector<OwningReductionGen> owningReductionGens;
  SmallVector<OwningAtomicReductionGen> owningAtomicReductionGens;
  SmallVector<llvm::OpenMPIRBuilder::ReductionInfo> reductionInfos;
  collectReductionInfo(op, builder, moduleTranslation, reductionDecls,
                       owningReductionGens, owningAtomicReductionGens,
                       privateReductionVariables, reductionInfos);

  // The call to createReductions below expects the block to have a
  // terminator. Create an unreachable instruction to serve as terminator
  // and remove it later.
  llvm::UnreachableInst *tempTerminator = builder.CreateUnreachable();
  builder.SetInsertPoint(tempTerminator);
  llvm::OpenMPIRBuilder::InsertPointTy contInsertPoint =
      ompBuilder->createReductions(builder.saveIP(), allocaIP, reductionInfos,
                                   isByRef, op.getNowait());
  if (!contInsertPoint.getBlock())
    return op->emitOpError() << "failed to convert reductions";
  auto nextInsertionPoint =
      ompBuilder->createBarrier(contInsertPoint, llvm::omp::OMPD_for);
  tempTerminator->eraseFromParent();
  builder.restoreIP(nextInsertionPoint);

  // after the construct, deallocate private reduction variables
  SmallVector<Region *> reductionRegions;
  llvm::transform(reductionDecls, std::back_inserter(reductionRegions),
                  [](omp::DeclareReductionOp reductionDecl) {
                    return &reductionDecl.getCleanupRegion();
                  });
  return inlineOmpRegionCleanup(reductionRegions, privateReductionVariables,
                                moduleTranslation, builder,
                                "omp.reduction.cleanup");
  return success();
}

static ArrayRef<bool> getIsByRef(std::optional<ArrayRef<bool>> attr) {
  if (!attr)
    return {};
  return *attr;
}

// TODO: not used by omp.parallel
template <typename OP>
static LogicalResult allocAndInitializeReductionVars(
    OP op, ArrayRef<BlockArgument> reductionArgs, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation,
    llvm::OpenMPIRBuilder::InsertPointTy &allocaIP,
    SmallVectorImpl<omp::DeclareReductionOp> &reductionDecls,
    SmallVectorImpl<llvm::Value *> &privateReductionVariables,
    DenseMap<Value, llvm::Value *> &reductionVariableMap,
    llvm::ArrayRef<bool> isByRef) {
  if (op.getNumReductionVars() == 0)
    return success();

  SmallVector<DeferredStore> deferredStores;

  if (failed(allocReductionVars(op, reductionArgs, builder, moduleTranslation,
                                allocaIP, reductionDecls,
                                privateReductionVariables, reductionVariableMap,
                                deferredStores, isByRef)))
    return failure();

  // store result of the alloc region to the allocated pointer to the real
  // reduction variable
  for (auto [data, addr] : deferredStores)
    builder.CreateStore(data, addr);

  // Before the loop, store the initial values of reductions into reduction
  // variables. Although this could be done after allocas, we don't want to mess
  // up with the alloca insertion point.
  for (unsigned i = 0; i < op.getNumReductionVars(); ++i) {
    SmallVector<llvm::Value *, 1> phis;

    // map block argument to initializer region
    mapInitializationArgs(op, moduleTranslation, reductionDecls,
                          reductionVariableMap, i);

    if (failed(inlineConvertOmpRegions(reductionDecls[i].getInitializerRegion(),
                                       "omp.reduction.neutral", builder,
                                       moduleTranslation, &phis)))
      return failure();
    assert(phis.size() == 1 && "expected one value to be yielded from the "
                               "reduction neutral element declaration region");
    if (isByRef[i]) {
      if (!reductionDecls[i].getAllocRegion().empty())
        // done in allocReductionVars
        continue;

      // TODO: this path can be removed once all users of by-ref are updated to
      // use an alloc region

      // Allocate reduction variable (which is a pointer to the real reduction
      // variable allocated in the inlined region)
      llvm::Value *var = builder.CreateAlloca(
          moduleTranslation.convertType(reductionDecls[i].getType()));
      // Store the result of the inlined region to the allocated reduction var
      // ptr
      builder.CreateStore(phis[0], var);

      privateReductionVariables[i] = var;
      moduleTranslation.mapValue(reductionArgs[i], phis[0]);
      reductionVariableMap.try_emplace(op.getReductionVars()[i], phis[0]);
    } else {
      // for by-ref case the store is inside of the reduction region
      builder.CreateStore(phis[0], privateReductionVariables[i]);
      // the rest was handled in allocByValReductionVars
    }

    // forget the mapping for the initializer region because we might need a
    // different mapping if this reduction declaration is re-used for a
    // different variable
    moduleTranslation.forgetMapping(reductionDecls[i].getInitializerRegion());
  }

  return success();
}

static LogicalResult
convertOmpSections(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  using StorableBodyGenCallbackTy =
      llvm::OpenMPIRBuilder::StorableBodyGenCallbackTy;

  auto sectionsOp = cast<omp::SectionsOp>(opInst);

  // TODO: Support the following clauses: private, firstprivate, lastprivate,
  // allocate
  if (!sectionsOp.getAllocateVars().empty() ||
      !sectionsOp.getAllocatorVars().empty() ||
      !sectionsOp.getPrivateVars().empty() || sectionsOp.getPrivateSyms())
    return opInst.emitError("unhandled clauses for translation to LLVM IR");

  llvm::ArrayRef<bool> isByRef = getIsByRef(sectionsOp.getReductionByref());
  assert(isByRef.size() == sectionsOp.getNumReductionVars());

  SmallVector<omp::DeclareReductionOp> reductionDecls;
  collectReductionDecls(sectionsOp, reductionDecls);
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  SmallVector<llvm::Value *> privateReductionVariables(
      sectionsOp.getNumReductionVars());
  DenseMap<Value, llvm::Value *> reductionVariableMap;

  MutableArrayRef<BlockArgument> reductionArgs =
      sectionsOp.getRegion().getArguments();

  if (failed(allocAndInitializeReductionVars(
          sectionsOp, reductionArgs, builder, moduleTranslation, allocaIP,
          reductionDecls, privateReductionVariables, reductionVariableMap,
          isByRef)))
    return failure();

  // Store the mapping between reduction variables and their private copies on
  // ModuleTranslation stack. It can be then recovered when translating
  // omp.reduce operations in a separate call.
  LLVM::ModuleTranslation::SaveStack<OpenMPVarMappingStackFrame> mappingGuard(
      moduleTranslation, reductionVariableMap);

  LogicalResult bodyGenStatus = success();
  SmallVector<StorableBodyGenCallbackTy> sectionCBs;

  for (Operation &op : *sectionsOp.getRegion().begin()) {
    auto sectionOp = dyn_cast<omp::SectionOp>(op);
    if (!sectionOp) // omp.terminator
      continue;

    Region &region = sectionOp.getRegion();
    auto sectionCB = [&sectionsOp, &region, &builder, &moduleTranslation,
                      &bodyGenStatus](InsertPointTy allocaIP,
                                      InsertPointTy codeGenIP) {
      builder.restoreIP(codeGenIP);

      // map the omp.section reduction block argument to the omp.sections block
      // arguments
      // TODO: this assumes that the only block arguments are reduction
      // variables
      assert(region.getNumArguments() ==
             sectionsOp.getRegion().getNumArguments());
      for (auto [sectionsArg, sectionArg] : llvm::zip_equal(
               sectionsOp.getRegion().getArguments(), region.getArguments())) {
        llvm::Value *llvmVal = moduleTranslation.lookupValue(sectionsArg);
        assert(llvmVal);
        moduleTranslation.mapValue(sectionArg, llvmVal);
      }

      convertOmpOpRegions(region, "omp.section.region", builder,
                          moduleTranslation, bodyGenStatus);
    };
    sectionCBs.push_back(sectionCB);
  }

  // No sections within omp.sections operation - skip generation. This situation
  // is only possible if there is only a terminator operation inside the
  // sections operation
  if (sectionCBs.empty())
    return success();

  assert(isa<omp::SectionOp>(*sectionsOp.getRegion().op_begin()));

  // TODO: Perform appropriate actions according to the data-sharing
  // attribute (shared, private, firstprivate, ...) of variables.
  // Currently defaults to shared.
  auto privCB = [&](InsertPointTy, InsertPointTy codeGenIP, llvm::Value &,
                    llvm::Value &vPtr,
                    llvm::Value *&replacementValue) -> InsertPointTy {
    replacementValue = &vPtr;
    return codeGenIP;
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createSections(
      ompLoc, allocaIP, sectionCBs, privCB, finiCB, false,
      sectionsOp.getNowait()));

  if (failed(bodyGenStatus))
    return bodyGenStatus;

  // Process the reductions if required.
  return createReductionsAndCleanup(sectionsOp, builder, moduleTranslation,
                                    allocaIP, reductionDecls,
                                    privateReductionVariables, isByRef);
}

/// Converts an OpenMP single construct into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpSingle(omp::SingleOp &singleOp, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  LogicalResult bodyGenStatus = success();
  if (!singleOp.getPrivateVars().empty() || singleOp.getPrivateSyms())
    return singleOp.emitError("unhandled clauses for translation to LLVM IR");

  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    builder.restoreIP(codegenIP);
    convertOmpOpRegions(singleOp.getRegion(), "omp.single.region", builder,
                        moduleTranslation, bodyGenStatus);
  };
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  // Handle copyprivate
  Operation::operand_range cpVars = singleOp.getCopyprivateVars();
  std::optional<ArrayAttr> cpFuncs = singleOp.getCopyprivateSyms();
  llvm::SmallVector<llvm::Value *> llvmCPVars;
  llvm::SmallVector<llvm::Function *> llvmCPFuncs;
  for (size_t i = 0, e = cpVars.size(); i < e; ++i) {
    llvmCPVars.push_back(moduleTranslation.lookupValue(cpVars[i]));
    auto llvmFuncOp = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
        singleOp, cast<SymbolRefAttr>((*cpFuncs)[i]));
    llvmCPFuncs.push_back(
        moduleTranslation.lookupFunction(llvmFuncOp.getName()));
  }

  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createSingle(
      ompLoc, bodyCB, finiCB, singleOp.getNowait(), llvmCPVars, llvmCPFuncs));
  return bodyGenStatus;
}

// Convert an OpenMP Teams construct to LLVM IR using OpenMPIRBuilder
static LogicalResult
convertOmpTeams(omp::TeamsOp op, llvm::IRBuilderBase &builder,
                LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  LogicalResult bodyGenStatus = success();
  if (!op.getAllocatorVars().empty() || op.getReductionSyms() ||
      !op.getPrivateVars().empty() || op.getPrivateSyms())
    return op.emitError("unhandled clauses for translation to LLVM IR");

  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
        moduleTranslation, allocaIP);
    builder.restoreIP(codegenIP);
    convertOmpOpRegions(op.getRegion(), "omp.teams.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  llvm::Value *numTeamsLower = nullptr;
  if (Value numTeamsLowerVar = op.getNumTeamsLower())
    numTeamsLower = moduleTranslation.lookupValue(numTeamsLowerVar);

  llvm::Value *numTeamsUpper = nullptr;
  if (Value numTeamsUpperVar = op.getNumTeamsUpper())
    numTeamsUpper = moduleTranslation.lookupValue(numTeamsUpperVar);

  llvm::Value *threadLimit = nullptr;
  if (Value threadLimitVar = op.getThreadLimit())
    threadLimit = moduleTranslation.lookupValue(threadLimitVar);

  llvm::Value *ifExpr = nullptr;
  if (Value ifVar = op.getIfExpr())
    ifExpr = moduleTranslation.lookupValue(ifVar);

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createTeams(
      ompLoc, bodyCB, numTeamsLower, numTeamsUpper, threadLimit, ifExpr));
  return bodyGenStatus;
}

static void
buildDependData(std::optional<ArrayAttr> dependKinds, OperandRange dependVars,
                LLVM::ModuleTranslation &moduleTranslation,
                SmallVectorImpl<llvm::OpenMPIRBuilder::DependData> &dds) {
  if (dependVars.empty())
    return;
  for (auto dep : llvm::zip(dependVars, dependKinds->getValue())) {
    llvm::omp::RTLDependenceKindTy type;
    switch (
        cast<mlir::omp::ClauseTaskDependAttr>(std::get<1>(dep)).getValue()) {
    case mlir::omp::ClauseTaskDepend::taskdependin:
      type = llvm::omp::RTLDependenceKindTy::DepIn;
      break;
    // The OpenMP runtime requires that the codegen for 'depend' clause for
    // 'out' dependency kind must be the same as codegen for 'depend' clause
    // with 'inout' dependency.
    case mlir::omp::ClauseTaskDepend::taskdependout:
    case mlir::omp::ClauseTaskDepend::taskdependinout:
      type = llvm::omp::RTLDependenceKindTy::DepInOut;
      break;
    };
    llvm::Value *depVal = moduleTranslation.lookupValue(std::get<0>(dep));
    llvm::OpenMPIRBuilder::DependData dd(type, depVal->getType(), depVal);
    dds.emplace_back(dd);
  }
}
/// Converts an OpenMP task construct into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpTaskOp(omp::TaskOp taskOp, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  LogicalResult bodyGenStatus = success();
  if (taskOp.getUntiedAttr() || taskOp.getMergeableAttr() ||
      taskOp.getInReductionSyms() || taskOp.getPriority() ||
      !taskOp.getAllocateVars().empty() || !taskOp.getPrivateVars().empty() ||
      taskOp.getPrivateSyms()) {
    return taskOp.emitError("unhandled clauses for translation to LLVM IR");
  }
  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    // Save the alloca insertion point on ModuleTranslation stack for use in
    // nested regions.
    LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
        moduleTranslation, allocaIP);

    builder.restoreIP(codegenIP);
    convertOmpOpRegions(taskOp.getRegion(), "omp.task.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  SmallVector<llvm::OpenMPIRBuilder::DependData> dds;
  buildDependData(taskOp.getDependKinds(), taskOp.getDependVars(),
                  moduleTranslation, dds);

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createTask(
      ompLoc, allocaIP, bodyCB, !taskOp.getUntied(),
      moduleTranslation.lookupValue(taskOp.getFinal()),
      moduleTranslation.lookupValue(taskOp.getIfExpr()), dds));
  return bodyGenStatus;
}

/// Converts an OpenMP taskgroup construct into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpTaskgroupOp(omp::TaskgroupOp tgOp, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  LogicalResult bodyGenStatus = success();
  if (!tgOp.getTaskReductionVars().empty() || !tgOp.getAllocateVars().empty()) {
    return tgOp.emitError("unhandled clauses for translation to LLVM IR");
  }
  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    builder.restoreIP(codegenIP);
    convertOmpOpRegions(tgOp.getRegion(), "omp.taskgroup.region", builder,
                        moduleTranslation, bodyGenStatus);
  };
  InsertPointTy allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createTaskgroup(
      ompLoc, allocaIP, bodyCB));
  return bodyGenStatus;
}

static LogicalResult
convertOmpTaskwaitOp(omp::TaskwaitOp twOp, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {
  if (!twOp.getDependVars().empty() || twOp.getDependKinds() ||
      twOp.getNowait())
    return twOp.emitError("unhandled clauses for translation to LLVM IR");

  moduleTranslation.getOpenMPBuilder()->createTaskwait(builder.saveIP());
  return success();
}

/// Converts an OpenMP workshare loop into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpWsloop(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  auto wsloopOp = cast<omp::WsloopOp>(opInst);
  if (!wsloopOp.getAllocateVars().empty() ||
      !wsloopOp.getAllocatorVars().empty() ||
      !wsloopOp.getPrivateVars().empty() || wsloopOp.getPrivateSyms())
    return opInst.emitError("unhandled clauses for translation to LLVM IR");

  // FIXME: Here any other nested wrappers (e.g. omp.simd) are skipped, so
  // codegen for composite constructs like 'DO/FOR SIMD' will be the same as for
  // 'DO/FOR'.
  auto loopOp = cast<omp::LoopNestOp>(wsloopOp.getWrappedLoop());

  llvm::ArrayRef<bool> isByRef = getIsByRef(wsloopOp.getReductionByref());
  assert(isByRef.size() == wsloopOp.getNumReductionVars());

  // Static is the default.
  auto schedule =
      wsloopOp.getScheduleKind().value_or(omp::ClauseScheduleKind::Static);

  // Find the loop configuration.
  llvm::Value *step = moduleTranslation.lookupValue(loopOp.getLoopSteps()[0]);
  llvm::Type *ivType = step->getType();
  llvm::Value *chunk = nullptr;
  if (wsloopOp.getScheduleChunk()) {
    llvm::Value *chunkVar =
        moduleTranslation.lookupValue(wsloopOp.getScheduleChunk());
    chunk = builder.CreateSExtOrTrunc(chunkVar, ivType);
  }

  SmallVector<omp::DeclareReductionOp> reductionDecls;
  collectReductionDecls(wsloopOp, reductionDecls);
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  SmallVector<llvm::Value *> privateReductionVariables(
      wsloopOp.getNumReductionVars());
  DenseMap<Value, llvm::Value *> reductionVariableMap;

  MutableArrayRef<BlockArgument> reductionArgs =
      wsloopOp.getRegion().getArguments();

  if (failed(allocAndInitializeReductionVars(
          wsloopOp, reductionArgs, builder, moduleTranslation, allocaIP,
          reductionDecls, privateReductionVariables, reductionVariableMap,
          isByRef)))
    return failure();

  // Store the mapping between reduction variables and their private copies on
  // ModuleTranslation stack. It can be then recovered when translating
  // omp.reduce operations in a separate call.
  LLVM::ModuleTranslation::SaveStack<OpenMPVarMappingStackFrame> mappingGuard(
      moduleTranslation, reductionVariableMap);

  // Set up the source location value for OpenMP runtime.
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  // Generator of the canonical loop body.
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  SmallVector<llvm::CanonicalLoopInfo *> loopInfos;
  SmallVector<llvm::OpenMPIRBuilder::InsertPointTy> bodyInsertPoints;
  LogicalResult bodyGenStatus = success();
  auto bodyGen = [&](llvm::OpenMPIRBuilder::InsertPointTy ip, llvm::Value *iv) {
    // Make sure further conversions know about the induction variable.
    moduleTranslation.mapValue(
        loopOp.getRegion().front().getArgument(loopInfos.size()), iv);

    // Capture the body insertion point for use in nested loops. BodyIP of the
    // CanonicalLoopInfo always points to the beginning of the entry block of
    // the body.
    bodyInsertPoints.push_back(ip);

    if (loopInfos.size() != loopOp.getNumLoops() - 1)
      return;

    // Convert the body of the loop.
    builder.restoreIP(ip);
    convertOmpOpRegions(loopOp.getRegion(), "omp.wsloop.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  // Delegate actual loop construction to the OpenMP IRBuilder.
  // TODO: this currently assumes omp.loop_nest is semantically similar to SCF
  // loop, i.e. it has a positive step, uses signed integer semantics.
  // Reconsider this code when the nested loop operation clearly supports more
  // cases.
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  for (unsigned i = 0, e = loopOp.getNumLoops(); i < e; ++i) {
    llvm::Value *lowerBound =
        moduleTranslation.lookupValue(loopOp.getLoopLowerBounds()[i]);
    llvm::Value *upperBound =
        moduleTranslation.lookupValue(loopOp.getLoopUpperBounds()[i]);
    llvm::Value *step = moduleTranslation.lookupValue(loopOp.getLoopSteps()[i]);

    // Make sure loop trip count are emitted in the preheader of the outermost
    // loop at the latest so that they are all available for the new collapsed
    // loop will be created below.
    llvm::OpenMPIRBuilder::LocationDescription loc = ompLoc;
    llvm::OpenMPIRBuilder::InsertPointTy computeIP = ompLoc.IP;
    if (i != 0) {
      loc = llvm::OpenMPIRBuilder::LocationDescription(bodyInsertPoints.back());
      computeIP = loopInfos.front()->getPreheaderIP();
    }
    loopInfos.push_back(ompBuilder->createCanonicalLoop(
        loc, bodyGen, lowerBound, upperBound, step,
        /*IsSigned=*/true, loopOp.getLoopInclusive(), computeIP));

    if (failed(bodyGenStatus))
      return failure();
  }

  // Collapse loops. Store the insertion point because LoopInfos may get
  // invalidated.
  llvm::IRBuilderBase::InsertPoint afterIP = loopInfos.front()->getAfterIP();
  llvm::CanonicalLoopInfo *loopInfo =
      ompBuilder->collapseLoops(ompLoc.DL, loopInfos, {});

  allocaIP = findAllocaInsertPoint(builder, moduleTranslation);

  // TODO: Handle doacross loops when the ordered clause has a parameter.
  bool isOrdered = wsloopOp.getOrdered().has_value();
  std::optional<omp::ScheduleModifier> scheduleMod = wsloopOp.getScheduleMod();
  bool isSimd = wsloopOp.getScheduleSimd();

  ompBuilder->applyWorkshareLoop(
      ompLoc.DL, loopInfo, allocaIP, !wsloopOp.getNowait(),
      convertToScheduleKind(schedule), chunk, isSimd,
      scheduleMod == omp::ScheduleModifier::monotonic,
      scheduleMod == omp::ScheduleModifier::nonmonotonic, isOrdered);

  // Continue building IR after the loop. Note that the LoopInfo returned by
  // `collapseLoops` points inside the outermost loop and is intended for
  // potential further loop transformations. Use the insertion point stored
  // before collapsing loops instead.
  builder.restoreIP(afterIP);

  // Process the reductions if required.
  return createReductionsAndCleanup(wsloopOp, builder, moduleTranslation,
                                    allocaIP, reductionDecls,
                                    privateReductionVariables, isByRef);
}

/// A RAII class that on construction replaces the region arguments of the
/// parallel op (which correspond to private variables) with the actual private
/// variables they correspond to. This prepares the parallel op so that it
/// matches what is expected by the OMPIRBuilder.
///
/// On destruction, it restores the original state of the operation so that on
/// the MLIR side, the op is not affected by conversion to LLVM IR.
class OmpParallelOpConversionManager {
public:
  OmpParallelOpConversionManager(omp::ParallelOp opInst)
      : region(opInst.getRegion()), privateVars(opInst.getPrivateVars()),
        privateArgBeginIdx(opInst.getNumReductionVars()),
        privateArgEndIdx(privateArgBeginIdx + privateVars.size()) {
    auto privateVarsIt = privateVars.begin();

    for (size_t argIdx = privateArgBeginIdx; argIdx < privateArgEndIdx;
         ++argIdx, ++privateVarsIt)
      mlir::replaceAllUsesInRegionWith(region.getArgument(argIdx),
                                       *privateVarsIt, region);
  }

  ~OmpParallelOpConversionManager() {
    auto privateVarsIt = privateVars.begin();

    for (size_t argIdx = privateArgBeginIdx; argIdx < privateArgEndIdx;
         ++argIdx, ++privateVarsIt)
      mlir::replaceAllUsesInRegionWith(*privateVarsIt,
                                       region.getArgument(argIdx), region);
  }

private:
  Region &region;
  OperandRange privateVars;
  unsigned privateArgBeginIdx;
  unsigned privateArgEndIdx;
};

/// Converts the OpenMP parallel operation to LLVM IR.
static LogicalResult
convertOmpParallel(omp::ParallelOp opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  OmpParallelOpConversionManager raii(opInst);
  ArrayRef<bool> isByRef = getIsByRef(opInst.getReductionByref());
  assert(isByRef.size() == opInst.getNumReductionVars());

  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // Collect reduction declarations
  SmallVector<omp::DeclareReductionOp> reductionDecls;
  collectReductionDecls(opInst, reductionDecls);
  SmallVector<llvm::Value *> privateReductionVariables(
      opInst.getNumReductionVars());
  SmallVector<DeferredStore> deferredStores;

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // Allocate reduction vars
    DenseMap<Value, llvm::Value *> reductionVariableMap;

    MutableArrayRef<BlockArgument> reductionArgs =
        opInst.getRegion().getArguments().slice(
            opInst.getNumAllocateVars() + opInst.getNumAllocatorsVars(),
            opInst.getNumReductionVars());

    allocaIP =
        InsertPointTy(allocaIP.getBlock(),
                      allocaIP.getBlock()->getTerminator()->getIterator());

    if (failed(allocReductionVars(
            opInst, reductionArgs, builder, moduleTranslation, allocaIP,
            reductionDecls, privateReductionVariables, reductionVariableMap,
            deferredStores, isByRef)))
      bodyGenStatus = failure();

    // Initialize reduction vars
    builder.restoreIP(allocaIP);
    llvm::BasicBlock *initBlock = splitBB(builder, true, "omp.reduction.init");
    allocaIP =
        InsertPointTy(allocaIP.getBlock(),
                      allocaIP.getBlock()->getTerminator()->getIterator());
    SmallVector<llvm::Value *> byRefVars(opInst.getNumReductionVars());
    for (unsigned i = 0; i < opInst.getNumReductionVars(); ++i) {
      if (isByRef[i]) {
        if (!reductionDecls[i].getAllocRegion().empty())
          continue;

        // TODO: remove after all users of by-ref are updated to use the alloc
        // region: Allocate reduction variable (which is a pointer to the real
        // reduciton variable allocated in the inlined region)
        byRefVars[i] = builder.CreateAlloca(
            moduleTranslation.convertType(reductionDecls[i].getType()));
      }
    }

    builder.SetInsertPoint(initBlock->getFirstNonPHIOrDbgOrAlloca());

    // insert stores deferred until after all allocas
    // these store the results of the alloc region into the allocation for the
    // pointer to the reduction variable
    for (auto [data, addr] : deferredStores)
      builder.CreateStore(data, addr);

    for (unsigned i = 0; i < opInst.getNumReductionVars(); ++i) {
      SmallVector<llvm::Value *> phis;

      // map the block argument
      mapInitializationArgs(opInst, moduleTranslation, reductionDecls,
                            reductionVariableMap, i);
      if (failed(inlineConvertOmpRegions(
              reductionDecls[i].getInitializerRegion(), "omp.reduction.neutral",
              builder, moduleTranslation, &phis)))
        bodyGenStatus = failure();
      assert(phis.size() == 1 &&
             "expected one value to be yielded from the "
             "reduction neutral element declaration region");

      builder.SetInsertPoint(builder.GetInsertBlock()->getTerminator());

      if (isByRef[i]) {
        if (!reductionDecls[i].getAllocRegion().empty())
          continue;

        // TODO: remove after all users of by-ref are updated to use the alloc

        // Store the result of the inlined region to the allocated reduction var
        // ptr
        builder.CreateStore(phis[0], byRefVars[i]);

        privateReductionVariables[i] = byRefVars[i];
        moduleTranslation.mapValue(reductionArgs[i], phis[0]);
        reductionVariableMap.try_emplace(opInst.getReductionVars()[i], phis[0]);
      } else {
        // for by-ref case the store is inside of the reduction init region
        builder.CreateStore(phis[0], privateReductionVariables[i]);
        // the rest is done in allocByValReductionVars
      }

      // clear block argument mapping in case it needs to be re-created with a
      // different source for another use of the same reduction decl
      moduleTranslation.forgetMapping(reductionDecls[i].getInitializerRegion());
    }

    // Store the mapping between reduction variables and their private copies on
    // ModuleTranslation stack. It can be then recovered when translating
    // omp.reduce operations in a separate call.
    LLVM::ModuleTranslation::SaveStack<OpenMPVarMappingStackFrame> mappingGuard(
        moduleTranslation, reductionVariableMap);

    // Save the alloca insertion point on ModuleTranslation stack for use in
    // nested regions.
    LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
        moduleTranslation, allocaIP);

    // ParallelOp has only one region associated with it.
    builder.restoreIP(codeGenIP);
    auto regionBlock =
        convertOmpOpRegions(opInst.getRegion(), "omp.par.region", builder,
                            moduleTranslation, bodyGenStatus);

    // Process the reductions if required.
    if (opInst.getNumReductionVars() > 0) {
      // Collect reduction info
      SmallVector<OwningReductionGen> owningReductionGens;
      SmallVector<OwningAtomicReductionGen> owningAtomicReductionGens;
      SmallVector<llvm::OpenMPIRBuilder::ReductionInfo> reductionInfos;
      collectReductionInfo(opInst, builder, moduleTranslation, reductionDecls,
                           owningReductionGens, owningAtomicReductionGens,
                           privateReductionVariables, reductionInfos);

      // Move to region cont block
      builder.SetInsertPoint(regionBlock->getTerminator());

      // Generate reductions from info
      llvm::UnreachableInst *tempTerminator = builder.CreateUnreachable();
      builder.SetInsertPoint(tempTerminator);

      llvm::OpenMPIRBuilder::InsertPointTy contInsertPoint =
          ompBuilder->createReductions(builder.saveIP(), allocaIP,
                                       reductionInfos, isByRef, false);
      if (!contInsertPoint.getBlock()) {
        bodyGenStatus = opInst->emitOpError() << "failed to convert reductions";
        return;
      }

      tempTerminator->eraseFromParent();
      builder.restoreIP(contInsertPoint);
    }
  };

  SmallVector<omp::PrivateClauseOp> mlirPrivatizerClones;
  SmallVector<llvm::Value *> llvmPrivateVars;

  // TODO: Perform appropriate actions according to the data-sharing
  // attribute (shared, private, firstprivate, ...) of variables.
  // Currently shared and private are supported.
  auto privCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                    llvm::Value &, llvm::Value &llvmOmpRegionInput,
                    llvm::Value *&llvmReplacementValue) -> InsertPointTy {
    llvmReplacementValue = &llvmOmpRegionInput;

    // If this is a private value, this lambda will return the corresponding
    // mlir value and its `PrivateClauseOp`. Otherwise, empty values are
    // returned.
    auto [mlirPrivVar, mlirPrivatizerClone] =
        [&]() -> std::pair<mlir::Value, omp::PrivateClauseOp> {
      if (!opInst.getPrivateVars().empty()) {
        auto mlirPrivVars = opInst.getPrivateVars();
        auto mlirPrivSyms = opInst.getPrivateSyms();

        // Try to find a privatizer that corresponds to the LLVM value being
        // privatized.
        for (auto [mlirPrivVar, mlirPrivatizerAttr] :
             llvm::zip_equal(mlirPrivVars, *mlirPrivSyms)) {
          // Find the MLIR private variable corresponding to the LLVM value
          // being privatized.
          llvm::Value *mlirToLLVMPrivVar =
              moduleTranslation.lookupValue(mlirPrivVar);

          // Check if the LLVM value being privatized matches the LLVM value
          // mapped to privVar. In some cases, this is not trivial ...
          auto isMatch = [&]() {
            if (mlirToLLVMPrivVar == nullptr)
              return false;

            // If both values are trivially equal, we found a match.
            if (mlirToLLVMPrivVar == &llvmOmpRegionInput)
              return true;

            // Otherwise, we check if both llvmOmpRegionInputPtr and
            // mlirToLLVMPrivVar refer to the same memory (through a load/store
            // pair). This happens if a struct (i.e. multi-field value) is being
            // privatized.
            //
            // For example, if the privatized value is defined by:
            // ```
            //   %priv_val = alloca { ptr, i64 }, align 8
            // ```
            //
            // The initialization of this value (outside the omp region) will be
            // something like this:
            //
            // clang-format off
            // ```
            //   %partially_init_priv_val = insertvalue { ptr, i64 } undef,
            //                              ptr %some_ptr, 0
            //   %fully_init_priv_val = insertvalue { ptr, i64 } %partially_init_priv_val,
            //                          i64 %some_i64, 1
            //   ...
            //   store { ptr, i64 } %fully_init_priv_val, ptr %priv_val, align 8
            // ```
            // clang-format on
            //
            // Now, we enter the OMP region, in order to access this privatized
            // value, we need to load from the allocated memory:
            // ```
            // omp.par.entry:
            //   %priv_val_load = load { ptr, i64 }, ptr %priv_val, align 8
            // ```
            //
            // The 2 LLVM values tracked here map as follows:
            // - `mlirToLLVMPrivVar`     -> `%fully_init_priv_val`
            // - `llvmOmpRegionInputPtr` -> `%priv_val_load`
            //
            // Even though they eventually refer to the same memory reference
            // (the memory being privatized), they are 2 distinct LLVM values.
            // Therefore, we need to discover their correspondence by finding
            // out if they store into and load from the same mem ref.
            auto *llvmOmpRegionInputPtrLoad =
                llvm::dyn_cast_if_present<llvm::LoadInst>(&llvmOmpRegionInput);

            if (llvmOmpRegionInputPtrLoad == nullptr)
              return false;

            for (auto &use : mlirToLLVMPrivVar->uses()) {
              auto *mlirToLLVMPrivVarStore =
                  llvm::dyn_cast_if_present<llvm::StoreInst>(use.getUser());
              if (mlirToLLVMPrivVarStore &&
                  (llvmOmpRegionInputPtrLoad->getPointerOperand() ==
                   mlirToLLVMPrivVarStore->getPointerOperand()))
                return true;
            }

            return false;
          };

          if (!isMatch())
            continue;

          SymbolRefAttr privSym = llvm::cast<SymbolRefAttr>(mlirPrivatizerAttr);
          omp::PrivateClauseOp privatizer =
              SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(
                  opInst, privSym);

          // Clone the privatizer in case it is used by more than one parallel
          // region. The privatizer is processed in-place (see below) before it
          // gets inlined in the parallel region and therefore processing the
          // original op is dangerous.

          MLIRContext &context = moduleTranslation.getContext();
          mlir::IRRewriter opCloner(&context);
          opCloner.setInsertionPoint(privatizer);
          auto clone = llvm::cast<mlir::omp::PrivateClauseOp>(
              opCloner.clone(*privatizer));

          // Unique the clone name to avoid clashes in the symbol table.
          unsigned counter = 0;
          SmallString<256> cloneName = SymbolTable::generateSymbolName<256>(
              privatizer.getSymName(),
              [&](llvm::StringRef candidate) {
                return SymbolTable::lookupNearestSymbolFrom(
                           opInst, StringAttr::get(&context, candidate)) !=
                       nullptr;
              },
              counter);

          clone.setSymName(cloneName);
          return {mlirPrivVar, clone};
        }
      }

      return {mlir::Value(), omp::PrivateClauseOp()};
    }();

    if (mlirPrivVar) {
      Region &allocRegion = mlirPrivatizerClone.getAllocRegion();

      // If this is a `firstprivate` clause, prepare the `omp.private` op by:
      if (mlirPrivatizerClone.getDataSharingType() ==
          omp::DataSharingClauseType::FirstPrivate) {
        auto oldAllocBackBlock = std::prev(allocRegion.end());
        omp::YieldOp oldAllocYieldOp =
            llvm::cast<omp::YieldOp>(oldAllocBackBlock->getTerminator());

        Region &copyRegion = mlirPrivatizerClone.getCopyRegion();

        mlir::IRRewriter copyCloneBuilder(&moduleTranslation.getContext());
        // 1. Cloning the `copy` region to the end of the `alloc` region.
        copyCloneBuilder.cloneRegionBefore(copyRegion, allocRegion,
                                           allocRegion.end());

        auto newCopyRegionFrontBlock = std::next(oldAllocBackBlock);
        // 2. Merging the last `alloc` block with the first block in the `copy`
        // region clone.
        // 3. Re-mapping the first argument of the `copy` region to be the
        // argument of the `alloc` region and the second argument of the `copy`
        // region to be the yielded value of the `alloc` region (this is the
        // private clone of the privatized value).
        copyCloneBuilder.mergeBlocks(
            &*newCopyRegionFrontBlock, &*oldAllocBackBlock,
            {allocRegion.getArgument(0), oldAllocYieldOp.getOperand(0)});

        // 4. The old terminator of the `alloc` region is not needed anymore, so
        // delete it.
        oldAllocYieldOp.erase();
      }

      // Replace the privatizer block argument with mlir value being privatized.
      // This way, the body of the privatizer will be changed from using the
      // region/block argument to the value being privatized.
      auto allocRegionArg = allocRegion.getArgument(0);
      replaceAllUsesInRegionWith(allocRegionArg, mlirPrivVar, allocRegion);

      auto oldIP = builder.saveIP();
      builder.restoreIP(allocaIP);

      SmallVector<llvm::Value *, 1> yieldedValues;
      if (failed(inlineConvertOmpRegions(allocRegion, "omp.privatizer", builder,
                                         moduleTranslation, &yieldedValues))) {
        opInst.emitError("failed to inline `alloc` region of an `omp.private` "
                         "op in the parallel region");
        bodyGenStatus = failure();
        mlirPrivatizerClone.erase();
      } else {
        assert(yieldedValues.size() == 1);
        llvmReplacementValue = yieldedValues.front();

        // Keep the LLVM replacement value and the op clone in case we need to
        // emit cleanup (i.e. deallocation) logic.
        llvmPrivateVars.push_back(llvmReplacementValue);
        mlirPrivatizerClones.push_back(mlirPrivatizerClone);
      }

      builder.restoreIP(oldIP);
    }

    return codeGenIP;
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {
    InsertPointTy oldIP = builder.saveIP();
    builder.restoreIP(codeGenIP);

    // if the reduction has a cleanup region, inline it here to finalize the
    // reduction variables
    SmallVector<Region *> reductionCleanupRegions;
    llvm::transform(reductionDecls, std::back_inserter(reductionCleanupRegions),
                    [](omp::DeclareReductionOp reductionDecl) {
                      return &reductionDecl.getCleanupRegion();
                    });
    if (failed(inlineOmpRegionCleanup(
            reductionCleanupRegions, privateReductionVariables,
            moduleTranslation, builder, "omp.reduction.cleanup")))
      bodyGenStatus = failure();

    SmallVector<Region *> privateCleanupRegions;
    llvm::transform(mlirPrivatizerClones,
                    std::back_inserter(privateCleanupRegions),
                    [](omp::PrivateClauseOp privatizer) {
                      return &privatizer.getDeallocRegion();
                    });

    if (failed(inlineOmpRegionCleanup(
            privateCleanupRegions, llvmPrivateVars, moduleTranslation, builder,
            "omp.private.dealloc", /*shouldLoadCleanupRegionArg=*/false)))
      bodyGenStatus = failure();

    builder.restoreIP(oldIP);
  };

  llvm::Value *ifCond = nullptr;
  if (auto ifVar = opInst.getIfExpr())
    ifCond = moduleTranslation.lookupValue(ifVar);
  llvm::Value *numThreads = nullptr;
  if (auto numThreadsVar = opInst.getNumThreads())
    numThreads = moduleTranslation.lookupValue(numThreadsVar);
  auto pbKind = llvm::omp::OMP_PROC_BIND_default;
  if (auto bind = opInst.getProcBindKind())
    pbKind = getProcBindKind(*bind);
  // TODO: Is the Parallel construct cancellable?
  bool isCancellable = false;

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  builder.restoreIP(
      ompBuilder->createParallel(ompLoc, allocaIP, bodyGenCB, privCB, finiCB,
                                 ifCond, numThreads, pbKind, isCancellable));

  for (mlir::omp::PrivateClauseOp privatizerClone : mlirPrivatizerClones)
    privatizerClone.erase();

  return bodyGenStatus;
}

/// Convert Order attribute to llvm::omp::OrderKind.
static llvm::omp::OrderKind
convertOrderKind(std::optional<omp::ClauseOrderKind> o) {
  if (!o)
    return llvm::omp::OrderKind::OMP_ORDER_unknown;
  switch (*o) {
  case omp::ClauseOrderKind::Concurrent:
    return llvm::omp::OrderKind::OMP_ORDER_concurrent;
  }
  llvm_unreachable("Unknown ClauseOrderKind kind");
}

/// Converts an OpenMP simd loop into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpSimd(Operation &opInst, llvm::IRBuilderBase &builder,
               LLVM::ModuleTranslation &moduleTranslation) {
  auto simdOp = cast<omp::SimdOp>(opInst);
  auto loopOp = cast<omp::LoopNestOp>(simdOp.getWrappedLoop());

  if (!simdOp.getLinearVars().empty() || !simdOp.getLinearStepVars().empty() ||
      !simdOp.getPrivateVars().empty() || simdOp.getPrivateSyms() ||
      !simdOp.getReductionVars().empty() || simdOp.getReductionByref() ||
      simdOp.getReductionSyms())
    return opInst.emitError("unhandled clauses for translation to LLVM IR");

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  // Generator of the canonical loop body.
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  SmallVector<llvm::CanonicalLoopInfo *> loopInfos;
  SmallVector<llvm::OpenMPIRBuilder::InsertPointTy> bodyInsertPoints;
  LogicalResult bodyGenStatus = success();
  auto bodyGen = [&](llvm::OpenMPIRBuilder::InsertPointTy ip, llvm::Value *iv) {
    // Make sure further conversions know about the induction variable.
    moduleTranslation.mapValue(
        loopOp.getRegion().front().getArgument(loopInfos.size()), iv);

    // Capture the body insertion point for use in nested loops. BodyIP of the
    // CanonicalLoopInfo always points to the beginning of the entry block of
    // the body.
    bodyInsertPoints.push_back(ip);

    if (loopInfos.size() != loopOp.getNumLoops() - 1)
      return;

    // Convert the body of the loop.
    builder.restoreIP(ip);
    convertOmpOpRegions(loopOp.getRegion(), "omp.simd.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  // Delegate actual loop construction to the OpenMP IRBuilder.
  // TODO: this currently assumes omp.loop_nest is semantically similar to SCF
  // loop, i.e. it has a positive step, uses signed integer semantics.
  // Reconsider this code when the nested loop operation clearly supports more
  // cases.
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  for (unsigned i = 0, e = loopOp.getNumLoops(); i < e; ++i) {
    llvm::Value *lowerBound =
        moduleTranslation.lookupValue(loopOp.getLoopLowerBounds()[i]);
    llvm::Value *upperBound =
        moduleTranslation.lookupValue(loopOp.getLoopUpperBounds()[i]);
    llvm::Value *step = moduleTranslation.lookupValue(loopOp.getLoopSteps()[i]);

    // Make sure loop trip count are emitted in the preheader of the outermost
    // loop at the latest so that they are all available for the new collapsed
    // loop will be created below.
    llvm::OpenMPIRBuilder::LocationDescription loc = ompLoc;
    llvm::OpenMPIRBuilder::InsertPointTy computeIP = ompLoc.IP;
    if (i != 0) {
      loc = llvm::OpenMPIRBuilder::LocationDescription(bodyInsertPoints.back(),
                                                       ompLoc.DL);
      computeIP = loopInfos.front()->getPreheaderIP();
    }
    loopInfos.push_back(ompBuilder->createCanonicalLoop(
        loc, bodyGen, lowerBound, upperBound, step,
        /*IsSigned=*/true, /*Inclusive=*/true, computeIP));

    if (failed(bodyGenStatus))
      return failure();
  }

  // Collapse loops.
  llvm::IRBuilderBase::InsertPoint afterIP = loopInfos.front()->getAfterIP();
  llvm::CanonicalLoopInfo *loopInfo =
      ompBuilder->collapseLoops(ompLoc.DL, loopInfos, {});

  llvm::ConstantInt *simdlen = nullptr;
  if (std::optional<uint64_t> simdlenVar = simdOp.getSimdlen())
    simdlen = builder.getInt64(simdlenVar.value());

  llvm::ConstantInt *safelen = nullptr;
  if (std::optional<uint64_t> safelenVar = simdOp.getSafelen())
    safelen = builder.getInt64(safelenVar.value());

  llvm::MapVector<llvm::Value *, llvm::Value *> alignedVars;
  llvm::omp::OrderKind order = convertOrderKind(simdOp.getOrder());
  ompBuilder->applySimd(loopInfo, alignedVars,
                        simdOp.getIfExpr()
                            ? moduleTranslation.lookupValue(simdOp.getIfExpr())
                            : nullptr,
                        order, simdlen, safelen);

  builder.restoreIP(afterIP);
  return success();
}

/// Convert an Atomic Ordering attribute to llvm::AtomicOrdering.
static llvm::AtomicOrdering
convertAtomicOrdering(std::optional<omp::ClauseMemoryOrderKind> ao) {
  if (!ao)
    return llvm::AtomicOrdering::Monotonic; // Default Memory Ordering

  switch (*ao) {
  case omp::ClauseMemoryOrderKind::Seq_cst:
    return llvm::AtomicOrdering::SequentiallyConsistent;
  case omp::ClauseMemoryOrderKind::Acq_rel:
    return llvm::AtomicOrdering::AcquireRelease;
  case omp::ClauseMemoryOrderKind::Acquire:
    return llvm::AtomicOrdering::Acquire;
  case omp::ClauseMemoryOrderKind::Release:
    return llvm::AtomicOrdering::Release;
  case omp::ClauseMemoryOrderKind::Relaxed:
    return llvm::AtomicOrdering::Monotonic;
  }
  llvm_unreachable("Unknown ClauseMemoryOrderKind kind");
}

/// Convert omp.atomic.read operation to LLVM IR.
static LogicalResult
convertOmpAtomicRead(Operation &opInst, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {

  auto readOp = cast<omp::AtomicReadOp>(opInst);
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  llvm::AtomicOrdering AO = convertAtomicOrdering(readOp.getMemoryOrder());
  llvm::Value *x = moduleTranslation.lookupValue(readOp.getX());
  llvm::Value *v = moduleTranslation.lookupValue(readOp.getV());

  llvm::Type *elementType =
      moduleTranslation.convertType(readOp.getElementType());

  llvm::OpenMPIRBuilder::AtomicOpValue V = {v, elementType, false, false};
  llvm::OpenMPIRBuilder::AtomicOpValue X = {x, elementType, false, false};
  builder.restoreIP(ompBuilder->createAtomicRead(ompLoc, X, V, AO));
  return success();
}

/// Converts an omp.atomic.write operation to LLVM IR.
static LogicalResult
convertOmpAtomicWrite(Operation &opInst, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  auto writeOp = cast<omp::AtomicWriteOp>(opInst);
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::AtomicOrdering ao = convertAtomicOrdering(writeOp.getMemoryOrder());
  llvm::Value *expr = moduleTranslation.lookupValue(writeOp.getExpr());
  llvm::Value *dest = moduleTranslation.lookupValue(writeOp.getX());
  llvm::Type *ty = moduleTranslation.convertType(writeOp.getExpr().getType());
  llvm::OpenMPIRBuilder::AtomicOpValue x = {dest, ty, /*isSigned=*/false,
                                            /*isVolatile=*/false};
  builder.restoreIP(ompBuilder->createAtomicWrite(ompLoc, x, expr, ao));
  return success();
}

/// Converts an LLVM dialect binary operation to the corresponding enum value
/// for `atomicrmw` supported binary operation.
llvm::AtomicRMWInst::BinOp convertBinOpToAtomic(Operation &op) {
  return llvm::TypeSwitch<Operation *, llvm::AtomicRMWInst::BinOp>(&op)
      .Case([&](LLVM::AddOp) { return llvm::AtomicRMWInst::BinOp::Add; })
      .Case([&](LLVM::SubOp) { return llvm::AtomicRMWInst::BinOp::Sub; })
      .Case([&](LLVM::AndOp) { return llvm::AtomicRMWInst::BinOp::And; })
      .Case([&](LLVM::OrOp) { return llvm::AtomicRMWInst::BinOp::Or; })
      .Case([&](LLVM::XOrOp) { return llvm::AtomicRMWInst::BinOp::Xor; })
      .Case([&](LLVM::UMaxOp) { return llvm::AtomicRMWInst::BinOp::UMax; })
      .Case([&](LLVM::UMinOp) { return llvm::AtomicRMWInst::BinOp::UMin; })
      .Case([&](LLVM::FAddOp) { return llvm::AtomicRMWInst::BinOp::FAdd; })
      .Case([&](LLVM::FSubOp) { return llvm::AtomicRMWInst::BinOp::FSub; })
      .Default(llvm::AtomicRMWInst::BinOp::BAD_BINOP);
}

/// Converts an OpenMP atomic update operation using OpenMPIRBuilder.
static LogicalResult
convertOmpAtomicUpdate(omp::AtomicUpdateOp &opInst,
                       llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // Convert values and types.
  auto &innerOpList = opInst.getRegion().front().getOperations();
  bool isXBinopExpr{false};
  llvm::AtomicRMWInst::BinOp binop;
  mlir::Value mlirExpr;
  llvm::Value *llvmExpr = nullptr;
  llvm::Value *llvmX = nullptr;
  llvm::Type *llvmXElementType = nullptr;
  if (innerOpList.size() == 2) {
    // The two operations here are the update and the terminator.
    // Since we can identify the update operation, there is a possibility
    // that we can generate the atomicrmw instruction.
    mlir::Operation &innerOp = *opInst.getRegion().front().begin();
    if (!llvm::is_contained(innerOp.getOperands(),
                            opInst.getRegion().getArgument(0))) {
      return opInst.emitError("no atomic update operation with region argument"
                              " as operand found inside atomic.update region");
    }
    binop = convertBinOpToAtomic(innerOp);
    isXBinopExpr = innerOp.getOperand(0) == opInst.getRegion().getArgument(0);
    mlirExpr = (isXBinopExpr ? innerOp.getOperand(1) : innerOp.getOperand(0));
    llvmExpr = moduleTranslation.lookupValue(mlirExpr);
  } else {
    // Since the update region includes more than one operation
    // we will resort to generating a cmpxchg loop.
    binop = llvm::AtomicRMWInst::BinOp::BAD_BINOP;
  }
  llvmX = moduleTranslation.lookupValue(opInst.getX());
  llvmXElementType = moduleTranslation.convertType(
      opInst.getRegion().getArgument(0).getType());
  llvm::OpenMPIRBuilder::AtomicOpValue llvmAtomicX = {llvmX, llvmXElementType,
                                                      /*isSigned=*/false,
                                                      /*isVolatile=*/false};

  llvm::AtomicOrdering atomicOrdering =
      convertAtomicOrdering(opInst.getMemoryOrder());

  // Generate update code.
  LogicalResult updateGenStatus = success();
  auto updateFn = [&opInst, &moduleTranslation, &updateGenStatus](
                      llvm::Value *atomicx,
                      llvm::IRBuilder<> &builder) -> llvm::Value * {
    Block &bb = *opInst.getRegion().begin();
    moduleTranslation.mapValue(*opInst.getRegion().args_begin(), atomicx);
    moduleTranslation.mapBlock(&bb, builder.GetInsertBlock());
    if (failed(moduleTranslation.convertBlock(bb, true, builder))) {
      updateGenStatus = (opInst.emitError()
                         << "unable to convert update operation to llvm IR");
      return nullptr;
    }
    omp::YieldOp yieldop = dyn_cast<omp::YieldOp>(bb.getTerminator());
    assert(yieldop && yieldop.getResults().size() == 1 &&
           "terminator must be omp.yield op and it must have exactly one "
           "argument");
    return moduleTranslation.lookupValue(yieldop.getResults()[0]);
  };

  // Handle ambiguous alloca, if any.
  auto allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(ompBuilder->createAtomicUpdate(
      ompLoc, allocaIP, llvmAtomicX, llvmExpr, atomicOrdering, binop, updateFn,
      isXBinopExpr));
  return updateGenStatus;
}

static LogicalResult
convertOmpAtomicCapture(omp::AtomicCaptureOp atomicCaptureOp,
                        llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  mlir::Value mlirExpr;
  bool isXBinopExpr = false, isPostfixUpdate = false;
  llvm::AtomicRMWInst::BinOp binop = llvm::AtomicRMWInst::BinOp::BAD_BINOP;

  omp::AtomicUpdateOp atomicUpdateOp = atomicCaptureOp.getAtomicUpdateOp();
  omp::AtomicWriteOp atomicWriteOp = atomicCaptureOp.getAtomicWriteOp();

  assert((atomicUpdateOp || atomicWriteOp) &&
         "internal op must be an atomic.update or atomic.write op");

  if (atomicWriteOp) {
    isPostfixUpdate = true;
    mlirExpr = atomicWriteOp.getExpr();
  } else {
    isPostfixUpdate = atomicCaptureOp.getSecondOp() ==
                      atomicCaptureOp.getAtomicUpdateOp().getOperation();
    auto &innerOpList = atomicUpdateOp.getRegion().front().getOperations();
    bool isRegionArgUsed{false};
    // Find the binary update operation that uses the region argument
    // and get the expression to update
    for (Operation &innerOp : innerOpList) {
      if (innerOp.getNumOperands() == 2) {
        binop = convertBinOpToAtomic(innerOp);
        if (!llvm::is_contained(innerOp.getOperands(),
                                atomicUpdateOp.getRegion().getArgument(0)))
          continue;
        isRegionArgUsed = true;
        isXBinopExpr =
            innerOp.getNumOperands() > 0 &&
            innerOp.getOperand(0) == atomicUpdateOp.getRegion().getArgument(0);
        mlirExpr =
            (isXBinopExpr ? innerOp.getOperand(1) : innerOp.getOperand(0));
        break;
      }
    }
    if (!isRegionArgUsed)
      return atomicUpdateOp.emitError(
          "no atomic update operation with region argument"
          " as operand found inside atomic.update region");
  }

  llvm::Value *llvmExpr = moduleTranslation.lookupValue(mlirExpr);
  llvm::Value *llvmX =
      moduleTranslation.lookupValue(atomicCaptureOp.getAtomicReadOp().getX());
  llvm::Value *llvmV =
      moduleTranslation.lookupValue(atomicCaptureOp.getAtomicReadOp().getV());
  llvm::Type *llvmXElementType = moduleTranslation.convertType(
      atomicCaptureOp.getAtomicReadOp().getElementType());
  llvm::OpenMPIRBuilder::AtomicOpValue llvmAtomicX = {llvmX, llvmXElementType,
                                                      /*isSigned=*/false,
                                                      /*isVolatile=*/false};
  llvm::OpenMPIRBuilder::AtomicOpValue llvmAtomicV = {llvmV, llvmXElementType,
                                                      /*isSigned=*/false,
                                                      /*isVolatile=*/false};

  llvm::AtomicOrdering atomicOrdering =
      convertAtomicOrdering(atomicCaptureOp.getMemoryOrder());

  LogicalResult updateGenStatus = success();
  auto updateFn = [&](llvm::Value *atomicx,
                      llvm::IRBuilder<> &builder) -> llvm::Value * {
    if (atomicWriteOp)
      return moduleTranslation.lookupValue(atomicWriteOp.getExpr());
    Block &bb = *atomicUpdateOp.getRegion().begin();
    moduleTranslation.mapValue(*atomicUpdateOp.getRegion().args_begin(),
                               atomicx);
    moduleTranslation.mapBlock(&bb, builder.GetInsertBlock());
    if (failed(moduleTranslation.convertBlock(bb, true, builder))) {
      updateGenStatus = (atomicUpdateOp.emitError()
                         << "unable to convert update operation to llvm IR");
      return nullptr;
    }
    omp::YieldOp yieldop = dyn_cast<omp::YieldOp>(bb.getTerminator());
    assert(yieldop && yieldop.getResults().size() == 1 &&
           "terminator must be omp.yield op and it must have exactly one "
           "argument");
    return moduleTranslation.lookupValue(yieldop.getResults()[0]);
  };

  // Handle ambiguous alloca, if any.
  auto allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(ompBuilder->createAtomicCapture(
      ompLoc, allocaIP, llvmAtomicX, llvmAtomicV, llvmExpr, atomicOrdering,
      binop, updateFn, atomicUpdateOp, isPostfixUpdate, isXBinopExpr));
  return updateGenStatus;
}

/// Converts an OpenMP Threadprivate operation into LLVM IR using
/// OpenMPIRBuilder.
static LogicalResult
convertOmpThreadprivate(Operation &opInst, llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  auto threadprivateOp = cast<omp::ThreadprivateOp>(opInst);

  Value symAddr = threadprivateOp.getSymAddr();
  auto *symOp = symAddr.getDefiningOp();
  if (!isa<LLVM::AddressOfOp>(symOp))
    return opInst.emitError("Addressing symbol not found");
  LLVM::AddressOfOp addressOfOp = dyn_cast<LLVM::AddressOfOp>(symOp);

  LLVM::GlobalOp global =
      addressOfOp.getGlobal(moduleTranslation.symbolTable());
  llvm::GlobalValue *globalValue = moduleTranslation.lookupGlobal(global);
  llvm::Type *type = globalValue->getValueType();
  llvm::TypeSize typeSize =
      builder.GetInsertBlock()->getModule()->getDataLayout().getTypeStoreSize(
          type);
  llvm::ConstantInt *size = builder.getInt64(typeSize.getFixedValue());
  llvm::StringRef suffix = llvm::StringRef(".cache", 6);
  std::string cacheName = (Twine(global.getSymName()).concat(suffix)).str();
  llvm::Value *callInst =
      moduleTranslation.getOpenMPBuilder()->createCachedThreadPrivate(
          ompLoc, globalValue, size, cacheName);
  moduleTranslation.mapValue(opInst.getResult(0), callInst);
  return success();
}

static llvm::OffloadEntriesInfoManager::OMPTargetDeviceClauseKind
convertToDeviceClauseKind(mlir::omp::DeclareTargetDeviceType deviceClause) {
  switch (deviceClause) {
  case mlir::omp::DeclareTargetDeviceType::host:
    return llvm::OffloadEntriesInfoManager::OMPTargetDeviceClauseHost;
    break;
  case mlir::omp::DeclareTargetDeviceType::nohost:
    return llvm::OffloadEntriesInfoManager::OMPTargetDeviceClauseNoHost;
    break;
  case mlir::omp::DeclareTargetDeviceType::any:
    return llvm::OffloadEntriesInfoManager::OMPTargetDeviceClauseAny;
    break;
  }
  llvm_unreachable("unhandled device clause");
}

static llvm::OffloadEntriesInfoManager::OMPTargetGlobalVarEntryKind
convertToCaptureClauseKind(
    mlir::omp::DeclareTargetCaptureClause captureClause) {
  switch (captureClause) {
  case mlir::omp::DeclareTargetCaptureClause::to:
    return llvm::OffloadEntriesInfoManager::OMPTargetGlobalVarEntryTo;
  case mlir::omp::DeclareTargetCaptureClause::link:
    return llvm::OffloadEntriesInfoManager::OMPTargetGlobalVarEntryLink;
  case mlir::omp::DeclareTargetCaptureClause::enter:
    return llvm::OffloadEntriesInfoManager::OMPTargetGlobalVarEntryEnter;
  }
  llvm_unreachable("unhandled capture clause");
}

static llvm::SmallString<64>
getDeclareTargetRefPtrSuffix(LLVM::GlobalOp globalOp,
                             llvm::OpenMPIRBuilder &ompBuilder) {
  llvm::SmallString<64> suffix;
  llvm::raw_svector_ostream os(suffix);
  if (globalOp.getVisibility() == mlir::SymbolTable::Visibility::Private) {
    auto loc = globalOp->getLoc()->findInstanceOf<FileLineColLoc>();
    auto fileInfoCallBack = [&loc]() {
      return std::pair<std::string, uint64_t>(
          llvm::StringRef(loc.getFilename()), loc.getLine());
    };

    os << llvm::format(
        "_%x", ompBuilder.getTargetEntryUniqueInfo(fileInfoCallBack).FileID);
  }
  os << "_decl_tgt_ref_ptr";

  return suffix;
}

static bool isDeclareTargetLink(mlir::Value value) {
  if (auto addressOfOp =
          llvm::dyn_cast_if_present<LLVM::AddressOfOp>(value.getDefiningOp())) {
    auto modOp = addressOfOp->getParentOfType<mlir::ModuleOp>();
    Operation *gOp = modOp.lookupSymbol(addressOfOp.getGlobalName());
    if (auto declareTargetGlobal =
            llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(gOp))
      if (declareTargetGlobal.getDeclareTargetCaptureClause() ==
          mlir::omp::DeclareTargetCaptureClause::link)
        return true;
  }
  return false;
}

// Returns the reference pointer generated by the lowering of the declare target
// operation in cases where the link clause is used or the to clause is used in
// USM mode.
static llvm::Value *
getRefPtrIfDeclareTarget(mlir::Value value,
                         LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // An easier way to do this may just be to keep track of any pointer
  // references and their mapping to their respective operation
  if (auto addressOfOp =
          llvm::dyn_cast_if_present<LLVM::AddressOfOp>(value.getDefiningOp())) {
    if (auto gOp = llvm::dyn_cast_or_null<LLVM::GlobalOp>(
            addressOfOp->getParentOfType<mlir::ModuleOp>().lookupSymbol(
                addressOfOp.getGlobalName()))) {

      if (auto declareTargetGlobal =
              llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
                  gOp.getOperation())) {

        // In this case, we must utilise the reference pointer generated by the
        // declare target operation, similar to Clang
        if ((declareTargetGlobal.getDeclareTargetCaptureClause() ==
             mlir::omp::DeclareTargetCaptureClause::link) ||
            (declareTargetGlobal.getDeclareTargetCaptureClause() ==
                 mlir::omp::DeclareTargetCaptureClause::to &&
             ompBuilder->Config.hasRequiresUnifiedSharedMemory())) {
          llvm::SmallString<64> suffix =
              getDeclareTargetRefPtrSuffix(gOp, *ompBuilder);

          if (gOp.getSymName().contains(suffix))
            return moduleTranslation.getLLVMModule()->getNamedValue(
                gOp.getSymName());

          return moduleTranslation.getLLVMModule()->getNamedValue(
              (gOp.getSymName().str() + suffix.str()).str());
        }
      }
    }
  }

  return nullptr;
}

namespace {
// A small helper structure to contain data gathered
// for map lowering and coalese it into one area and
// avoiding extra computations such as searches in the
// llvm module for lowered mapped variables or checking
// if something is declare target (and retrieving the
// value) more than neccessary.
struct MapInfoData : llvm::OpenMPIRBuilder::MapInfosTy {
  llvm::SmallVector<bool, 4> IsDeclareTarget;
  llvm::SmallVector<bool, 4> IsAMember;
  // Identify if mapping was added by mapClause or use_device clauses.
  llvm::SmallVector<bool, 4> IsAMapping;
  llvm::SmallVector<mlir::Operation *, 4> MapClause;
  llvm::SmallVector<llvm::Value *, 4> OriginalValue;
  // Stripped off array/pointer to get the underlying
  // element type
  llvm::SmallVector<llvm::Type *, 4> BaseType;

  /// Append arrays in \a CurInfo.
  void append(MapInfoData &CurInfo) {
    IsDeclareTarget.append(CurInfo.IsDeclareTarget.begin(),
                           CurInfo.IsDeclareTarget.end());
    MapClause.append(CurInfo.MapClause.begin(), CurInfo.MapClause.end());
    OriginalValue.append(CurInfo.OriginalValue.begin(),
                         CurInfo.OriginalValue.end());
    BaseType.append(CurInfo.BaseType.begin(), CurInfo.BaseType.end());
    llvm::OpenMPIRBuilder::MapInfosTy::append(CurInfo);
  }
};
} // namespace

uint64_t getArrayElementSizeInBits(LLVM::LLVMArrayType arrTy, DataLayout &dl) {
  if (auto nestedArrTy = llvm::dyn_cast_if_present<LLVM::LLVMArrayType>(
          arrTy.getElementType()))
    return getArrayElementSizeInBits(nestedArrTy, dl);
  return dl.getTypeSizeInBits(arrTy.getElementType());
}

// This function calculates the size to be offloaded for a specified type, given
// its associated map clause (which can contain bounds information which affects
// the total size), this size is calculated based on the underlying element type
// e.g. given a 1-D array of ints, we will calculate the size from the integer
// type * number of elements in the array. This size can be used in other
// calculations but is ultimately used as an argument to the OpenMP runtimes
// kernel argument structure which is generated through the combinedInfo data
// structures.
// This function is somewhat equivalent to Clang's getExprTypeSize inside of
// CGOpenMPRuntime.cpp.
llvm::Value *getSizeInBytes(DataLayout &dl, const mlir::Type &type,
                            Operation *clauseOp, llvm::Value *basePointer,
                            llvm::Type *baseType, llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation) {
  if (auto memberClause =
          mlir::dyn_cast_if_present<mlir::omp::MapInfoOp>(clauseOp)) {
    // This calculates the size to transfer based on bounds and the underlying
    // element type, provided bounds have been specified (Fortran
    // pointers/allocatables/target and arrays that have sections specified fall
    // into this as well).
    if (!memberClause.getBounds().empty()) {
      llvm::Value *elementCount = builder.getInt64(1);
      for (auto bounds : memberClause.getBounds()) {
        if (auto boundOp = mlir::dyn_cast_if_present<mlir::omp::MapBoundsOp>(
                bounds.getDefiningOp())) {
          // The below calculation for the size to be mapped calculated from the
          // map.info's bounds is: (elemCount * [UB - LB] + 1), later we
          // multiply by the underlying element types byte size to get the full
          // size to be offloaded based on the bounds
          elementCount = builder.CreateMul(
              elementCount,
              builder.CreateAdd(
                  builder.CreateSub(
                      moduleTranslation.lookupValue(boundOp.getUpperBound()),
                      moduleTranslation.lookupValue(boundOp.getLowerBound())),
                  builder.getInt64(1)));
        }
      }

      // utilising getTypeSizeInBits instead of getTypeSize as getTypeSize gives
      // the size in inconsistent byte or bit format.
      uint64_t underlyingTypeSzInBits = dl.getTypeSizeInBits(type);
      if (auto arrTy = llvm::dyn_cast_if_present<LLVM::LLVMArrayType>(type))
        underlyingTypeSzInBits = getArrayElementSizeInBits(arrTy, dl);

      // The size in bytes x number of elements, the sizeInBytes stored is
      // the underyling types size, e.g. if ptr<i32>, it'll be the i32's
      // size, so we do some on the fly runtime math to get the size in
      // bytes from the extent (ub - lb) * sizeInBytes. NOTE: This may need
      // some adjustment for members with more complex types.
      return builder.CreateMul(elementCount,
                               builder.getInt64(underlyingTypeSzInBits / 8));
    }
  }

  return builder.getInt64(dl.getTypeSizeInBits(type) / 8);
}

static void collectMapDataFromMapOperands(
    MapInfoData &mapData, SmallVectorImpl<Value> &mapVars,
    LLVM::ModuleTranslation &moduleTranslation, DataLayout &dl,
    llvm::IRBuilderBase &builder, const ArrayRef<Value> &useDevPtrOperands = {},
    const ArrayRef<Value> &useDevAddrOperands = {}) {
  auto checkIsAMember = [](const auto &mapVars, auto mapOp) {
    // Check if this is a member mapping and correctly assign that it is, if
    // it is a member of a larger object.
    // TODO: Need better handling of members, and distinguishing of members
    // that are implicitly allocated on device vs explicitly passed in as
    // arguments.
    // TODO: May require some further additions to support nested record
    // types, i.e. member maps that can have member maps.
    for (Value mapValue : mapVars) {
      auto map = cast<omp::MapInfoOp>(mapValue.getDefiningOp());
      for (auto member : map.getMembers())
        if (member == mapOp)
          return true;
    }
    return false;
  };

  // Process MapOperands
  for (Value mapValue : mapVars) {
    auto mapOp = cast<omp::MapInfoOp>(mapValue.getDefiningOp());
    Value offloadPtr =
        mapOp.getVarPtrPtr() ? mapOp.getVarPtrPtr() : mapOp.getVarPtr();
    mapData.OriginalValue.push_back(moduleTranslation.lookupValue(offloadPtr));
    mapData.Pointers.push_back(mapData.OriginalValue.back());

    if (llvm::Value *refPtr =
            getRefPtrIfDeclareTarget(offloadPtr,
                                     moduleTranslation)) { // declare target
      mapData.IsDeclareTarget.push_back(true);
      mapData.BasePointers.push_back(refPtr);
    } else { // regular mapped variable
      mapData.IsDeclareTarget.push_back(false);
      mapData.BasePointers.push_back(mapData.OriginalValue.back());
    }

    mapData.BaseType.push_back(
        moduleTranslation.convertType(mapOp.getVarType()));
    mapData.Sizes.push_back(
        getSizeInBytes(dl, mapOp.getVarType(), mapOp, mapData.Pointers.back(),
                       mapData.BaseType.back(), builder, moduleTranslation));
    mapData.MapClause.push_back(mapOp.getOperation());
    mapData.Types.push_back(
        llvm::omp::OpenMPOffloadMappingFlags(mapOp.getMapType().value()));
    mapData.Names.push_back(LLVM::createMappingInformation(
        mapOp.getLoc(), *moduleTranslation.getOpenMPBuilder()));
    mapData.DevicePointers.push_back(llvm::OpenMPIRBuilder::DeviceInfoTy::None);
    mapData.IsAMapping.push_back(true);
    mapData.IsAMember.push_back(checkIsAMember(mapVars, mapOp));
  }

  auto findMapInfo = [&mapData](llvm::Value *val,
                                llvm::OpenMPIRBuilder::DeviceInfoTy devInfoTy) {
    unsigned index = 0;
    bool found = false;
    for (llvm::Value *basePtr : mapData.OriginalValue) {
      if (basePtr == val && mapData.IsAMapping[index]) {
        found = true;
        mapData.Types[index] |=
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_RETURN_PARAM;
        mapData.DevicePointers[index] = devInfoTy;
      }
      index++;
    }
    return found;
  };

  // Process useDevPtr(Addr)Operands
  auto addDevInfos = [&](const llvm::ArrayRef<Value> &useDevOperands,
                         llvm::OpenMPIRBuilder::DeviceInfoTy devInfoTy) {
    for (Value mapValue : useDevOperands) {
      auto mapOp = cast<omp::MapInfoOp>(mapValue.getDefiningOp());
      Value offloadPtr =
          mapOp.getVarPtrPtr() ? mapOp.getVarPtrPtr() : mapOp.getVarPtr();
      llvm::Value *origValue = moduleTranslation.lookupValue(offloadPtr);

      // Check if map info is already present for this entry.
      if (!findMapInfo(origValue, devInfoTy)) {
        mapData.OriginalValue.push_back(origValue);
        mapData.Pointers.push_back(mapData.OriginalValue.back());
        mapData.IsDeclareTarget.push_back(false);
        mapData.BasePointers.push_back(mapData.OriginalValue.back());
        mapData.BaseType.push_back(
            moduleTranslation.convertType(mapOp.getVarType()));
        mapData.Sizes.push_back(builder.getInt64(0));
        mapData.MapClause.push_back(mapOp.getOperation());
        mapData.Types.push_back(
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_RETURN_PARAM);
        mapData.Names.push_back(LLVM::createMappingInformation(
            mapOp.getLoc(), *moduleTranslation.getOpenMPBuilder()));
        mapData.DevicePointers.push_back(devInfoTy);
        mapData.IsAMapping.push_back(true);
        mapData.IsAMember.push_back(checkIsAMember(useDevOperands, mapOp));
      }
    }
  };

  addDevInfos(useDevPtrOperands, llvm::OpenMPIRBuilder::DeviceInfoTy::Pointer);
  addDevInfos(useDevAddrOperands, llvm::OpenMPIRBuilder::DeviceInfoTy::Address);
}

static int getMapDataMemberIdx(MapInfoData &mapData, omp::MapInfoOp memberOp) {
  auto *res = llvm::find(mapData.MapClause, memberOp);
  assert(res != mapData.MapClause.end() &&
         "MapInfoOp for member not found in MapData, cannot return index");
  return std::distance(mapData.MapClause.begin(), res);
}

static omp::MapInfoOp getFirstOrLastMappedMemberPtr(omp::MapInfoOp mapInfo,
                                                    bool first) {
  DenseIntElementsAttr indexAttr = mapInfo.getMembersIndexAttr();

  // Only 1 member has been mapped, we can return it.
  if (indexAttr.size() == 1)
    if (auto mapOp =
            dyn_cast<omp::MapInfoOp>(mapInfo.getMembers()[0].getDefiningOp()))
      return mapOp;

  llvm::ArrayRef<int64_t> shape = indexAttr.getShapedType().getShape();
  llvm::SmallVector<size_t> indices(shape[0]);
  std::iota(indices.begin(), indices.end(), 0);

  llvm::sort(indices.begin(), indices.end(),
             [&](const size_t a, const size_t b) {
               auto indexValues = indexAttr.getValues<int32_t>();
               for (int i = 0; i < shape[1]; ++i) {
                 int aIndex = indexValues[a * shape[1] + i];
                 int bIndex = indexValues[b * shape[1] + i];

                 if (aIndex == bIndex)
                   continue;

                 if (aIndex != -1 && bIndex == -1)
                   return false;

                 if (aIndex == -1 && bIndex != -1)
                   return true;

                 // A is earlier in the record type layout than B
                 if (aIndex < bIndex)
                   return first;

                 if (bIndex < aIndex)
                   return !first;
               }

               // Iterated the entire list and couldn't make a decision, all
               // elements were likely the same. Return false, since the sort
               // comparator should return false for equal elements.
               return false;
             });

  return llvm::cast<omp::MapInfoOp>(
      mapInfo.getMembers()[indices.front()].getDefiningOp());
}

/// This function calculates the array/pointer offset for map data provided
/// with bounds operations, e.g. when provided something like the following:
///
/// Fortran
///     map(tofrom: array(2:5, 3:2))
///   or
/// C++
///   map(tofrom: array[1:4][2:3])
/// We must calculate the initial pointer offset to pass across, this function
/// performs this using bounds.
///
/// NOTE: which while specified in row-major order it currently needs to be
/// flipped for Fortran's column order array allocation and access (as
/// opposed to C++'s row-major, hence the backwards processing where order is
/// important). This is likely important to keep in mind for the future when
/// we incorporate a C++ frontend, both frontends will need to agree on the
/// ordering of generated bounds operations (one may have to flip them) to
/// make the below lowering frontend agnostic. The offload size
/// calcualtion may also have to be adjusted for C++.
std::vector<llvm::Value *>
calculateBoundsOffset(LLVM::ModuleTranslation &moduleTranslation,
                      llvm::IRBuilderBase &builder, bool isArrayTy,
                      OperandRange bounds) {
  std::vector<llvm::Value *> idx;
  // There's no bounds to calculate an offset from, we can safely
  // ignore and return no indices.
  if (bounds.empty())
    return idx;

  // If we have an array type, then we have its type so can treat it as a
  // normal GEP instruction where the bounds operations are simply indexes
  // into the array. We currently do reverse order of the bounds, which
  // I believe leans more towards Fortran's column-major in memory.
  if (isArrayTy) {
    idx.push_back(builder.getInt64(0));
    for (int i = bounds.size() - 1; i >= 0; --i) {
      if (auto boundOp = dyn_cast_if_present<omp::MapBoundsOp>(
              bounds[i].getDefiningOp())) {
        idx.push_back(moduleTranslation.lookupValue(boundOp.getLowerBound()));
      }
    }
  } else {
    // If we do not have an array type, but we have bounds, then we're dealing
    // with a pointer that's being treated like an array and we have the
    // underlying type e.g. an i32, or f64 etc, e.g. a fortran descriptor base
    // address (pointer pointing to the actual data) so we must caclulate the
    // offset using a single index which the following two loops attempts to
    // compute.

    // Calculates the size offset we need to make per row e.g. first row or
    // column only needs to be offset by one, but the next would have to be
    // the previous row/column offset multiplied by the extent of current row.
    //
    // For example ([1][10][100]):
    //
    //  - First row/column we move by 1 for each index increment
    //  - Second row/column we move by 1 (first row/column) * 10 (extent/size of
    //  current) for 10 for each index increment
    //  - Third row/column we would move by 10 (second row/column) *
    //  (extent/size of current) 100 for 1000 for each index increment
    std::vector<llvm::Value *> dimensionIndexSizeOffset{builder.getInt64(1)};
    for (size_t i = 1; i < bounds.size(); ++i) {
      if (auto boundOp = dyn_cast_if_present<omp::MapBoundsOp>(
              bounds[i].getDefiningOp())) {
        dimensionIndexSizeOffset.push_back(builder.CreateMul(
            moduleTranslation.lookupValue(boundOp.getExtent()),
            dimensionIndexSizeOffset[i - 1]));
      }
    }

    // Now that we have calculated how much we move by per index, we must
    // multiply each lower bound offset in indexes by the size offset we
    // have calculated in the previous and accumulate the results to get
    // our final resulting offset.
    for (int i = bounds.size() - 1; i >= 0; --i) {
      if (auto boundOp = dyn_cast_if_present<omp::MapBoundsOp>(
              bounds[i].getDefiningOp())) {
        if (idx.empty())
          idx.emplace_back(builder.CreateMul(
              moduleTranslation.lookupValue(boundOp.getLowerBound()),
              dimensionIndexSizeOffset[i]));
        else
          idx.back() = builder.CreateAdd(
              idx.back(), builder.CreateMul(moduleTranslation.lookupValue(
                                                boundOp.getLowerBound()),
                                            dimensionIndexSizeOffset[i]));
      }
    }
  }

  return idx;
}

// This creates two insertions into the MapInfosTy data structure for the
// "parent" of a set of members, (usually a container e.g.
// class/structure/derived type) when subsequent members have also been
// explicitly mapped on the same map clause. Certain types, such as Fortran
// descriptors are mapped like this as well, however, the members are
// implicit as far as a user is concerned, but we must explicitly map them
// internally.
//
// This function also returns the memberOfFlag for this particular parent,
// which is utilised in subsequent member mappings (by modifying there map type
// with it) to indicate that a member is part of this parent and should be
// treated by the runtime as such. Important to achieve the correct mapping.
//
// This function borrows a lot from Clang's emitCombinedEntry function
// inside of CGOpenMPRuntime.cpp
static llvm::omp::OpenMPOffloadMappingFlags mapParentWithMembers(
    LLVM::ModuleTranslation &moduleTranslation, llvm::IRBuilderBase &builder,
    llvm::OpenMPIRBuilder &ompBuilder, DataLayout &dl,
    llvm::OpenMPIRBuilder::MapInfosTy &combinedInfo, MapInfoData &mapData,
    uint64_t mapDataIndex, bool isTargetParams) {
  // Map the first segment of our structure
  combinedInfo.Types.emplace_back(
      isTargetParams
          ? llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TARGET_PARAM
          : llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE);
  combinedInfo.DevicePointers.emplace_back(
      mapData.DevicePointers[mapDataIndex]);
  combinedInfo.Names.emplace_back(LLVM::createMappingInformation(
      mapData.MapClause[mapDataIndex]->getLoc(), ompBuilder));
  combinedInfo.BasePointers.emplace_back(mapData.BasePointers[mapDataIndex]);

  // Calculate size of the parent object being mapped based on the
  // addresses at runtime, highAddr - lowAddr = size. This of course
  // doesn't factor in allocated data like pointers, hence the further
  // processing of members specified by users, or in the case of
  // Fortran pointers and allocatables, the mapping of the pointed to
  // data by the descriptor (which itself, is a structure containing
  // runtime information on the dynamically allocated data).
  auto parentClause =
      llvm::cast<omp::MapInfoOp>(mapData.MapClause[mapDataIndex]);

  llvm::Value *lowAddr, *highAddr;
  if (!parentClause.getPartialMap()) {
    lowAddr = builder.CreatePointerCast(mapData.Pointers[mapDataIndex],
                                        builder.getPtrTy());
    highAddr = builder.CreatePointerCast(
        builder.CreateConstGEP1_32(mapData.BaseType[mapDataIndex],
                                   mapData.Pointers[mapDataIndex], 1),
        builder.getPtrTy());
    combinedInfo.Pointers.emplace_back(mapData.Pointers[mapDataIndex]);
  } else {
    auto mapOp = dyn_cast<omp::MapInfoOp>(mapData.MapClause[mapDataIndex]);
    int firstMemberIdx = getMapDataMemberIdx(
        mapData, getFirstOrLastMappedMemberPtr(mapOp, true));
    lowAddr = builder.CreatePointerCast(mapData.Pointers[firstMemberIdx],
                                        builder.getPtrTy());
    int lastMemberIdx = getMapDataMemberIdx(
        mapData, getFirstOrLastMappedMemberPtr(mapOp, false));
    highAddr = builder.CreatePointerCast(
        builder.CreateGEP(mapData.BaseType[lastMemberIdx],
                          mapData.Pointers[lastMemberIdx], builder.getInt64(1)),
        builder.getPtrTy());
    combinedInfo.Pointers.emplace_back(mapData.Pointers[firstMemberIdx]);
  }

  llvm::Value *size = builder.CreateIntCast(
      builder.CreatePtrDiff(builder.getInt8Ty(), highAddr, lowAddr),
      builder.getInt64Ty(),
      /*isSigned=*/false);
  combinedInfo.Sizes.push_back(size);

  // TODO: This will need to be expanded to include the whole host of logic for
  // the map flags that Clang currently supports (e.g. it should take the map
  // flag of the parent map flag, remove the OMP_MAP_TARGET_PARAM and do some
  // further case specific flag modifications). For the moment, it handles what
  // we support as expected.
  llvm::omp::OpenMPOffloadMappingFlags mapFlag =
      llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;

  llvm::omp::OpenMPOffloadMappingFlags memberOfFlag =
      ompBuilder.getMemberOfFlag(combinedInfo.BasePointers.size() - 1);
  ompBuilder.setCorrectMemberOfFlag(mapFlag, memberOfFlag);

  // This creates the initial MEMBER_OF mapping that consists of
  // the parent/top level container (same as above effectively, except
  // with a fixed initial compile time size and separate maptype which
  // indicates the true mape type (tofrom etc.). This parent mapping is
  // only relevant if the structure in its totality is being mapped,
  // otherwise the above suffices.
  if (!parentClause.getPartialMap()) {
    combinedInfo.Types.emplace_back(mapFlag);
    combinedInfo.DevicePointers.emplace_back(
        llvm::OpenMPIRBuilder::DeviceInfoTy::None);
    combinedInfo.Names.emplace_back(LLVM::createMappingInformation(
        mapData.MapClause[mapDataIndex]->getLoc(), ompBuilder));
    combinedInfo.BasePointers.emplace_back(mapData.BasePointers[mapDataIndex]);
    combinedInfo.Pointers.emplace_back(mapData.Pointers[mapDataIndex]);
    combinedInfo.Sizes.emplace_back(mapData.Sizes[mapDataIndex]);
  }
  return memberOfFlag;
}

// The intent is to verify if the mapped data being passed is a
// pointer -> pointee that requires special handling in certain cases,
// e.g. applying the OMP_MAP_PTR_AND_OBJ map type.
//
// There may be a better way to verify this, but unfortunately with
// opaque pointers we lose the ability to easily check if something is
// a pointer whilst maintaining access to the underlying type.
static bool checkIfPointerMap(omp::MapInfoOp mapOp) {
  // If we have a varPtrPtr field assigned then the underlying type is a pointer
  if (mapOp.getVarPtrPtr())
    return true;

  // If the map data is declare target with a link clause, then it's represented
  // as a pointer when we lower it to LLVM-IR even if at the MLIR level it has
  // no relation to pointers.
  if (isDeclareTargetLink(mapOp.getVarPtr()))
    return true;

  return false;
}

// This function is intended to add explicit mappings of members
static void processMapMembersWithParent(
    LLVM::ModuleTranslation &moduleTranslation, llvm::IRBuilderBase &builder,
    llvm::OpenMPIRBuilder &ompBuilder, DataLayout &dl,
    llvm::OpenMPIRBuilder::MapInfosTy &combinedInfo, MapInfoData &mapData,
    uint64_t mapDataIndex, llvm::omp::OpenMPOffloadMappingFlags memberOfFlag) {

  auto parentClause =
      llvm::cast<omp::MapInfoOp>(mapData.MapClause[mapDataIndex]);

  for (auto mappedMembers : parentClause.getMembers()) {
    auto memberClause =
        llvm::cast<omp::MapInfoOp>(mappedMembers.getDefiningOp());
    int memberDataIdx = getMapDataMemberIdx(mapData, memberClause);

    assert(memberDataIdx >= 0 && "could not find mapped member of structure");

    // Same MemberOfFlag to indicate its link with parent and other members
    // of.
    auto mapFlag =
        llvm::omp::OpenMPOffloadMappingFlags(memberClause.getMapType().value());
    mapFlag &= ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TARGET_PARAM;
    mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_MEMBER_OF;
    ompBuilder.setCorrectMemberOfFlag(mapFlag, memberOfFlag);
    if (checkIfPointerMap(memberClause))
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PTR_AND_OBJ;

    combinedInfo.Types.emplace_back(mapFlag);
    combinedInfo.DevicePointers.emplace_back(
        mapData.DevicePointers[memberDataIdx]);
    combinedInfo.Names.emplace_back(
        LLVM::createMappingInformation(memberClause.getLoc(), ompBuilder));
    combinedInfo.BasePointers.emplace_back(mapData.BasePointers[mapDataIndex]);
    combinedInfo.Pointers.emplace_back(mapData.Pointers[memberDataIdx]);
    combinedInfo.Sizes.emplace_back(mapData.Sizes[memberDataIdx]);
  }
}

static void
processIndividualMap(MapInfoData &mapData, size_t mapDataIdx,
                     llvm::OpenMPIRBuilder::MapInfosTy &combinedInfo,
                     bool isTargetParams, int mapDataParentIdx = -1) {
  // Declare Target Mappings are excluded from being marked as
  // OMP_MAP_TARGET_PARAM as they are not passed as parameters, they're
  // marked with OMP_MAP_PTR_AND_OBJ instead.
  auto mapFlag = mapData.Types[mapDataIdx];
  auto mapInfoOp = llvm::cast<omp::MapInfoOp>(mapData.MapClause[mapDataIdx]);

  bool isPtrTy = checkIfPointerMap(mapInfoOp);
  if (isPtrTy)
    mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PTR_AND_OBJ;

  if (isTargetParams && !mapData.IsDeclareTarget[mapDataIdx])
    mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TARGET_PARAM;

  if (mapInfoOp.getMapCaptureType().value() ==
          omp::VariableCaptureKind::ByCopy &&
      !isPtrTy)
    mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_LITERAL;

  // if we're provided a mapDataParentIdx, then the data being mapped is
  // part of a larger object (in a parent <-> member mapping) and in this
  // case our BasePointer should be the parent.
  if (mapDataParentIdx >= 0)
    combinedInfo.BasePointers.emplace_back(
        mapData.BasePointers[mapDataParentIdx]);
  else
    combinedInfo.BasePointers.emplace_back(mapData.BasePointers[mapDataIdx]);

  combinedInfo.Pointers.emplace_back(mapData.Pointers[mapDataIdx]);
  combinedInfo.DevicePointers.emplace_back(mapData.DevicePointers[mapDataIdx]);
  combinedInfo.Names.emplace_back(mapData.Names[mapDataIdx]);
  combinedInfo.Types.emplace_back(mapFlag);
  combinedInfo.Sizes.emplace_back(mapData.Sizes[mapDataIdx]);
}

static void processMapWithMembersOf(
    LLVM::ModuleTranslation &moduleTranslation, llvm::IRBuilderBase &builder,
    llvm::OpenMPIRBuilder &ompBuilder, DataLayout &dl,
    llvm::OpenMPIRBuilder::MapInfosTy &combinedInfo, MapInfoData &mapData,
    uint64_t mapDataIndex, bool isTargetParams) {
  auto parentClause =
      llvm::cast<omp::MapInfoOp>(mapData.MapClause[mapDataIndex]);

  // If we have a partial map (no parent referenced in the map clauses of the
  // directive, only members) and only a single member, we do not need to bind
  // the map of the member to the parent, we can pass the member separately.
  if (parentClause.getMembers().size() == 1 && parentClause.getPartialMap()) {
    auto memberClause = llvm::cast<omp::MapInfoOp>(
        parentClause.getMembers()[0].getDefiningOp());
    int memberDataIdx = getMapDataMemberIdx(mapData, memberClause);
    // Note: Clang treats arrays with explicit bounds that fall into this
    // category as a parent with map case, however, it seems this isn't a
    // requirement, and processing them as an individual map is fine. So,
    // we will handle them as individual maps for the moment, as it's
    // difficult for us to check this as we always require bounds to be
    // specified currently and it's also marginally more optimal (single
    // map rather than two). The difference may come from the fact that
    // Clang maps array without bounds as pointers (which we do not
    // currently do), whereas we treat them as arrays in all cases
    // currently.
    processIndividualMap(mapData, memberDataIdx, combinedInfo, isTargetParams,
                         mapDataIndex);
    return;
  }

  llvm::omp::OpenMPOffloadMappingFlags memberOfParentFlag =
      mapParentWithMembers(moduleTranslation, builder, ompBuilder, dl,
                           combinedInfo, mapData, mapDataIndex, isTargetParams);
  processMapMembersWithParent(moduleTranslation, builder, ompBuilder, dl,
                              combinedInfo, mapData, mapDataIndex,
                              memberOfParentFlag);
}

// This is a variation on Clang's GenerateOpenMPCapturedVars, which
// generates different operation (e.g. load/store) combinations for
// arguments to the kernel, based on map capture kinds which are then
// utilised in the combinedInfo in place of the original Map value.
static void
createAlteredByCaptureMap(MapInfoData &mapData,
                          LLVM::ModuleTranslation &moduleTranslation,
                          llvm::IRBuilderBase &builder) {
  for (size_t i = 0; i < mapData.MapClause.size(); ++i) {
    // if it's declare target, skip it, it's handled separately.
    if (!mapData.IsDeclareTarget[i]) {
      auto mapOp = cast<omp::MapInfoOp>(mapData.MapClause[i]);
      omp::VariableCaptureKind captureKind =
          mapOp.getMapCaptureType().value_or(omp::VariableCaptureKind::ByRef);
      bool isPtrTy = checkIfPointerMap(mapOp);

      // Currently handles array sectioning lowerbound case, but more
      // logic may be required in the future. Clang invokes EmitLValue,
      // which has specialised logic for special Clang types such as user
      // defines, so it is possible we will have to extend this for
      // structures or other complex types. As the general idea is that this
      // function mimics some of the logic from Clang that we require for
      // kernel argument passing from host -> device.
      switch (captureKind) {
      case omp::VariableCaptureKind::ByRef: {
        llvm::Value *newV = mapData.Pointers[i];
        std::vector<llvm::Value *> offsetIdx = calculateBoundsOffset(
            moduleTranslation, builder, mapData.BaseType[i]->isArrayTy(),
            mapOp.getBounds());
        if (isPtrTy)
          newV = builder.CreateLoad(builder.getPtrTy(), newV);

        if (!offsetIdx.empty())
          newV = builder.CreateInBoundsGEP(mapData.BaseType[i], newV, offsetIdx,
                                           "array_offset");
        mapData.Pointers[i] = newV;
      } break;
      case omp::VariableCaptureKind::ByCopy: {
        llvm::Type *type = mapData.BaseType[i];
        llvm::Value *newV;
        if (mapData.Pointers[i]->getType()->isPointerTy())
          newV = builder.CreateLoad(type, mapData.Pointers[i]);
        else
          newV = mapData.Pointers[i];

        if (!isPtrTy) {
          auto curInsert = builder.saveIP();
          builder.restoreIP(findAllocaInsertPoint(builder, moduleTranslation));
          auto *memTempAlloc =
              builder.CreateAlloca(builder.getPtrTy(), nullptr, ".casted");
          builder.restoreIP(curInsert);

          builder.CreateStore(newV, memTempAlloc);
          newV = builder.CreateLoad(builder.getPtrTy(), memTempAlloc);
        }

        mapData.Pointers[i] = newV;
        mapData.BasePointers[i] = newV;
      } break;
      case omp::VariableCaptureKind::This:
      case omp::VariableCaptureKind::VLAType:
        mapData.MapClause[i]->emitOpError("Unhandled capture kind");
        break;
      }
    }
  }
}

// Generate all map related information and fill the combinedInfo.
static void genMapInfos(llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation,
                        DataLayout &dl,
                        llvm::OpenMPIRBuilder::MapInfosTy &combinedInfo,
                        MapInfoData &mapData, bool isTargetParams = false) {
  // We wish to modify some of the methods in which arguments are
  // passed based on their capture type by the target region, this can
  // involve generating new loads and stores, which changes the
  // MLIR value to LLVM value mapping, however, we only wish to do this
  // locally for the current function/target and also avoid altering
  // ModuleTranslation, so we remap the base pointer or pointer stored
  // in the map infos corresponding MapInfoData, which is later accessed
  // by genMapInfos and createTarget to help generate the kernel and
  // kernel arg structure. It primarily becomes relevant in cases like
  // bycopy, or byref range'd arrays. In the default case, we simply
  // pass thee pointer byref as both basePointer and pointer.
  if (!moduleTranslation.getOpenMPBuilder()->Config.isTargetDevice())
    createAlteredByCaptureMap(mapData, moduleTranslation, builder);

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // We operate under the assumption that all vectors that are
  // required in MapInfoData are of equal lengths (either filled with
  // default constructed data or appropiate information) so we can
  // utilise the size from any component of MapInfoData, if we can't
  // something is missing from the initial MapInfoData construction.
  for (size_t i = 0; i < mapData.MapClause.size(); ++i) {
    // NOTE/TODO: We currently do not support arbitrary depth record
    // type mapping.
    if (mapData.IsAMember[i])
      continue;

    auto mapInfoOp = dyn_cast<omp::MapInfoOp>(mapData.MapClause[i]);
    if (!mapInfoOp.getMembers().empty()) {
      processMapWithMembersOf(moduleTranslation, builder, *ompBuilder, dl,
                              combinedInfo, mapData, i, isTargetParams);
      continue;
    }

    processIndividualMap(mapData, i, combinedInfo, isTargetParams);
  }
}

static LogicalResult
convertOmpTargetData(Operation *op, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *ifCond = nullptr;
  int64_t deviceID = llvm::omp::OMP_DEVICEID_UNDEF;
  SmallVector<Value> mapVars;
  SmallVector<Value> useDevicePtrVars;
  SmallVector<Value> useDeviceAddrVars;
  llvm::omp::RuntimeFunction RTLFn;
  DataLayout DL = DataLayout(op->getParentOfType<ModuleOp>());

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  LogicalResult result =
      llvm::TypeSwitch<Operation *, LogicalResult>(op)
          .Case([&](omp::TargetDataOp dataOp) {
            if (auto ifVar = dataOp.getIfExpr())
              ifCond = moduleTranslation.lookupValue(ifVar);

            if (auto devId = dataOp.getDevice())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(devId.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  deviceID = intAttr.getInt();

            mapVars = dataOp.getMapVars();
            useDevicePtrVars = dataOp.getUseDevicePtrVars();
            useDeviceAddrVars = dataOp.getUseDeviceAddrVars();
            return success();
          })
          .Case([&](omp::TargetEnterDataOp enterDataOp) {
            if (enterDataOp.getNowait())
              return (LogicalResult)(enterDataOp.emitError(
                  "`nowait` is not supported yet"));

            if (auto ifVar = enterDataOp.getIfExpr())
              ifCond = moduleTranslation.lookupValue(ifVar);

            if (auto devId = enterDataOp.getDevice())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(devId.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  deviceID = intAttr.getInt();
            RTLFn = llvm::omp::OMPRTL___tgt_target_data_begin_mapper;
            mapVars = enterDataOp.getMapVars();
            return success();
          })
          .Case([&](omp::TargetExitDataOp exitDataOp) {
            if (exitDataOp.getNowait())
              return (LogicalResult)(exitDataOp.emitError(
                  "`nowait` is not supported yet"));

            if (auto ifVar = exitDataOp.getIfExpr())
              ifCond = moduleTranslation.lookupValue(ifVar);

            if (auto devId = exitDataOp.getDevice())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(devId.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  deviceID = intAttr.getInt();

            RTLFn = llvm::omp::OMPRTL___tgt_target_data_end_mapper;
            mapVars = exitDataOp.getMapVars();
            return success();
          })
          .Case([&](omp::TargetUpdateOp updateDataOp) {
            if (updateDataOp.getNowait())
              return (LogicalResult)(updateDataOp.emitError(
                  "`nowait` is not supported yet"));

            if (auto ifVar = updateDataOp.getIfExpr())
              ifCond = moduleTranslation.lookupValue(ifVar);

            if (auto devId = updateDataOp.getDevice())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(devId.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  deviceID = intAttr.getInt();

            RTLFn = llvm::omp::OMPRTL___tgt_target_data_update_mapper;
            mapVars = updateDataOp.getMapVars();
            return success();
          })
          .Default([&](Operation *op) {
            return op->emitError("unsupported OpenMP operation: ")
                   << op->getName();
          });

  if (failed(result))
    return failure();

  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;

  MapInfoData mapData;
  collectMapDataFromMapOperands(mapData, mapVars, moduleTranslation, DL,
                                builder, useDevicePtrVars, useDeviceAddrVars);

  // Fill up the arrays with all the mapped variables.
  llvm::OpenMPIRBuilder::MapInfosTy combinedInfo;
  auto genMapInfoCB =
      [&](InsertPointTy codeGenIP) -> llvm::OpenMPIRBuilder::MapInfosTy & {
    builder.restoreIP(codeGenIP);
    genMapInfos(builder, moduleTranslation, DL, combinedInfo, mapData);
    return combinedInfo;
  };

  llvm::OpenMPIRBuilder::TargetDataInfo info(/*RequiresDevicePointerInfo=*/true,
                                             /*SeparateBeginEndCalls=*/true);

  using BodyGenTy = llvm::OpenMPIRBuilder::BodyGenTy;
  LogicalResult bodyGenStatus = success();
  auto bodyGenCB = [&](InsertPointTy codeGenIP, BodyGenTy bodyGenType) {
    assert(isa<omp::TargetDataOp>(op) &&
           "BodyGen requested for non TargetDataOp");
    Region &region = cast<omp::TargetDataOp>(op).getRegion();
    switch (bodyGenType) {
    case BodyGenTy::Priv:
      // Check if any device ptr/addr info is available
      if (!info.DevicePtrInfoMap.empty()) {
        builder.restoreIP(codeGenIP);
        unsigned argIndex = 0;
        for (auto [basePointer, devicePointer] : llvm::zip_equal(
                 combinedInfo.BasePointers, combinedInfo.DevicePointers)) {
          if (devicePointer == llvm::OpenMPIRBuilder::DeviceInfoTy::Pointer) {
            const auto &arg = region.front().getArgument(argIndex);
            moduleTranslation.mapValue(
                arg, info.DevicePtrInfoMap[basePointer].second);
            argIndex++;
          } else if (devicePointer ==
                     llvm::OpenMPIRBuilder::DeviceInfoTy::Address) {
            const auto &arg = region.front().getArgument(argIndex);
            auto *loadInst = builder.CreateLoad(
                builder.getPtrTy(), info.DevicePtrInfoMap[basePointer].second);
            moduleTranslation.mapValue(arg, loadInst);
            argIndex++;
          }
        }

        bodyGenStatus = inlineConvertOmpRegions(region, "omp.data.region",
                                                builder, moduleTranslation);
      }
      break;
    case BodyGenTy::DupNoPriv:
      break;
    case BodyGenTy::NoPriv:
      // If device info is available then region has already been generated
      if (info.DevicePtrInfoMap.empty()) {
        builder.restoreIP(codeGenIP);
        // For device pass, if use_device_ptr(addr) mappings were present,
        // we need to link them here before codegen.
        if (ompBuilder->Config.IsTargetDevice.value_or(false)) {
          unsigned argIndex = 0;
          for (auto [basePointer, devicePointer] :
               llvm::zip_equal(mapData.BasePointers, mapData.DevicePointers)) {
            if (devicePointer == llvm::OpenMPIRBuilder::DeviceInfoTy::Pointer ||
                devicePointer == llvm::OpenMPIRBuilder::DeviceInfoTy::Address) {
              const auto &arg = region.front().getArgument(argIndex);
              moduleTranslation.mapValue(arg, basePointer);
              argIndex++;
            }
          }
        }
        bodyGenStatus = inlineConvertOmpRegions(region, "omp.data.region",
                                                builder, moduleTranslation);
      }
      break;
    }
    return builder.saveIP();
  };

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  if (isa<omp::TargetDataOp>(op)) {
    builder.restoreIP(ompBuilder->createTargetData(
        ompLoc, allocaIP, builder.saveIP(), builder.getInt64(deviceID), ifCond,
        info, genMapInfoCB, nullptr, bodyGenCB));
  } else {
    builder.restoreIP(ompBuilder->createTargetData(
        ompLoc, allocaIP, builder.saveIP(), builder.getInt64(deviceID), ifCond,
        info, genMapInfoCB, &RTLFn));
  }

  return bodyGenStatus;
}

/// Lowers the FlagsAttr which is applied to the module on the device
/// pass when offloading, this attribute contains OpenMP RTL globals that can
/// be passed as flags to the frontend, otherwise they are set to default
LogicalResult convertFlagsAttr(Operation *op, mlir::omp::FlagsAttr attribute,
                               LLVM::ModuleTranslation &moduleTranslation) {
  if (!cast<mlir::ModuleOp>(op))
    return failure();

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  ompBuilder->M.addModuleFlag(llvm::Module::Max, "openmp-device",
                              attribute.getOpenmpDeviceVersion());

  if (attribute.getNoGpuLib())
    return success();

  ompBuilder->createGlobalFlag(
      attribute.getDebugKind() /*LangOpts().OpenMPTargetDebug*/,
      "__omp_rtl_debug_kind");
  ompBuilder->createGlobalFlag(
      attribute
          .getAssumeTeamsOversubscription() /*LangOpts().OpenMPTeamSubscription*/
      ,
      "__omp_rtl_assume_teams_oversubscription");
  ompBuilder->createGlobalFlag(
      attribute
          .getAssumeThreadsOversubscription() /*LangOpts().OpenMPThreadSubscription*/
      ,
      "__omp_rtl_assume_threads_oversubscription");
  ompBuilder->createGlobalFlag(
      attribute.getAssumeNoThreadState() /*LangOpts().OpenMPNoThreadState*/,
      "__omp_rtl_assume_no_thread_state");
  ompBuilder->createGlobalFlag(
      attribute
          .getAssumeNoNestedParallelism() /*LangOpts().OpenMPNoNestedParallelism*/
      ,
      "__omp_rtl_assume_no_nested_parallelism");
  return success();
}

static bool getTargetEntryUniqueInfo(llvm::TargetRegionEntryInfo &targetInfo,
                                     omp::TargetOp targetOp,
                                     llvm::StringRef parentName = "") {
  auto fileLoc = targetOp.getLoc()->findInstanceOf<FileLineColLoc>();

  assert(fileLoc && "No file found from location");
  StringRef fileName = fileLoc.getFilename().getValue();

  llvm::sys::fs::UniqueID id;
  if (auto ec = llvm::sys::fs::getUniqueID(fileName, id)) {
    targetOp.emitError("Unable to get unique ID for file");
    return false;
  }

  uint64_t line = fileLoc.getLine();
  targetInfo = llvm::TargetRegionEntryInfo(parentName, id.getDevice(),
                                           id.getFile(), line);
  return true;
}

static bool targetOpSupported(Operation &opInst) {
  auto targetOp = cast<omp::TargetOp>(opInst);
  if (targetOp.getIfExpr()) {
    opInst.emitError("If clause not yet supported");
    return false;
  }

  if (targetOp.getDevice()) {
    opInst.emitError("Device clause not yet supported");
    return false;
  }

  if (targetOp.getThreadLimit()) {
    opInst.emitError("Thread limit clause not yet supported");
    return false;
  }

  if (targetOp.getNowait()) {
    opInst.emitError("Nowait clause not yet supported");
    return false;
  }

  if (!targetOp.getAllocateVars().empty() ||
      !targetOp.getAllocatorVars().empty()) {
    opInst.emitError("Allocate clause not yet supported");
    return false;
  }

  if (!targetOp.getInReductionVars().empty() ||
      targetOp.getInReductionByref() || targetOp.getInReductionSyms()) {
    opInst.emitError("In reduction clause not yet supported");
    return false;
  }

  return true;
}

static void
handleDeclareTargetMapVar(MapInfoData &mapData,
                          LLVM::ModuleTranslation &moduleTranslation,
                          llvm::IRBuilderBase &builder, llvm::Function *func) {
  for (size_t i = 0; i < mapData.MapClause.size(); ++i) {
    // In the case of declare target mapped variables, the basePointer is
    // the reference pointer generated by the convertDeclareTargetAttr
    // method. Whereas the kernelValue is the original variable, so for
    // the device we must replace all uses of this original global variable
    // (stored in kernelValue) with the reference pointer (stored in
    // basePointer for declare target mapped variables), as for device the
    // data is mapped into this reference pointer and should be loaded
    // from it, the original variable is discarded. On host both exist and
    // metadata is generated (elsewhere in the convertDeclareTargetAttr)
    // function to link the two variables in the runtime and then both the
    // reference pointer and the pointer are assigned in the kernel argument
    // structure for the host.
    if (mapData.IsDeclareTarget[i]) {
      // If the original map value is a constant, then we have to make sure all
      // of it's uses within the current kernel/function that we are going to
      // rewrite are converted to instructions, as we will be altering the old
      // use (OriginalValue) from a constant to an instruction, which will be
      // illegal and ICE the compiler if the user is a constant expression of
      // some kind e.g. a constant GEP.
      if (auto *constant = dyn_cast<llvm::Constant>(mapData.OriginalValue[i]))
        convertUsersOfConstantsToInstructions(constant, func, false);

      // The users iterator will get invalidated if we modify an element,
      // so we populate this vector of uses to alter each user on an
      // individual basis to emit its own load (rather than one load for
      // all).
      llvm::SmallVector<llvm::User *> userVec;
      for (llvm::User *user : mapData.OriginalValue[i]->users())
        userVec.push_back(user);

      for (llvm::User *user : userVec) {
        if (auto *insn = dyn_cast<llvm::Instruction>(user)) {
          if (insn->getFunction() == func) {
            auto *load = builder.CreateLoad(mapData.BasePointers[i]->getType(),
                                            mapData.BasePointers[i]);
            load->moveBefore(insn);
            user->replaceUsesOfWith(mapData.OriginalValue[i], load);
          }
        }
      }
    }
  }
}

// The createDeviceArgumentAccessor function generates
// instructions for retrieving (acessing) kernel
// arguments inside of the device kernel for use by
// the kernel. This enables different semantics such as
// the creation of temporary copies of data allowing
// semantics like read-only/no host write back kernel
// arguments.
//
// This currently implements a very light version of Clang's
// EmitParmDecl's handling of direct argument handling as well
// as a portion of the argument access generation based on
// capture types found at the end of emitOutlinedFunctionPrologue
// in Clang. The indirect path handling of EmitParmDecl's may be
// required for future work, but a direct 1-to-1 copy doesn't seem
// possible as the logic is rather scattered throughout Clang's
// lowering and perhaps we wish to deviate slightly.
//
// \param mapData - A container containing vectors of information
// corresponding to the input argument, which should have a
// corresponding entry in the MapInfoData containers
// OrigialValue's.
// \param arg - This is the generated kernel function argument that
// corresponds to the passed in input argument. We generated different
// accesses of this Argument, based on capture type and other Input
// related information.
// \param input - This is the host side value that will be passed to
// the kernel i.e. the kernel input, we rewrite all uses of this within
// the kernel (as we generate the kernel body based on the target's region
// which maintians references to the original input) to the retVal argument
// apon exit of this function inside of the OMPIRBuilder. This interlinks
// the kernel argument to future uses of it in the function providing
// appropriate "glue" instructions inbetween.
// \param retVal - This is the value that all uses of input inside of the
// kernel will be re-written to, the goal of this function is to generate
// an appropriate location for the kernel argument to be accessed from,
// e.g. ByRef will result in a temporary allocation location and then
// a store of the kernel argument into this allocated memory which
// will then be loaded from, ByCopy will use the allocated memory
// directly.
static llvm::IRBuilderBase::InsertPoint
createDeviceArgumentAccessor(MapInfoData &mapData, llvm::Argument &arg,
                             llvm::Value *input, llvm::Value *&retVal,
                             llvm::IRBuilderBase &builder,
                             llvm::OpenMPIRBuilder &ompBuilder,
                             LLVM::ModuleTranslation &moduleTranslation,
                             llvm::IRBuilderBase::InsertPoint allocaIP,
                             llvm::IRBuilderBase::InsertPoint codeGenIP) {
  builder.restoreIP(allocaIP);

  omp::VariableCaptureKind capture = omp::VariableCaptureKind::ByRef;

  // Find the associated MapInfoData entry for the current input
  for (size_t i = 0; i < mapData.MapClause.size(); ++i)
    if (mapData.OriginalValue[i] == input) {
      auto mapOp = cast<omp::MapInfoOp>(mapData.MapClause[i]);
      capture =
          mapOp.getMapCaptureType().value_or(omp::VariableCaptureKind::ByRef);

      break;
    }

  unsigned int allocaAS = ompBuilder.M.getDataLayout().getAllocaAddrSpace();
  unsigned int defaultAS =
      ompBuilder.M.getDataLayout().getProgramAddressSpace();

  // Create the alloca for the argument the current point.
  llvm::Value *v = builder.CreateAlloca(arg.getType(), allocaAS);

  if (allocaAS != defaultAS && arg.getType()->isPointerTy())
    v = builder.CreatePointerBitCastOrAddrSpaceCast(
        v, arg.getType()->getPointerTo(defaultAS));

  builder.CreateStore(&arg, v);

  builder.restoreIP(codeGenIP);

  switch (capture) {
  case omp::VariableCaptureKind::ByCopy: {
    retVal = v;
    break;
  }
  case omp::VariableCaptureKind::ByRef: {
    retVal = builder.CreateAlignedLoad(
        v->getType(), v,
        ompBuilder.M.getDataLayout().getPrefTypeAlign(v->getType()));
    break;
  }
  case omp::VariableCaptureKind::This:
  case omp::VariableCaptureKind::VLAType:
    assert(false && "Currently unsupported capture kind");
    break;
  }

  return builder.saveIP();
}

static LogicalResult
convertOmpTarget(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {

  if (!targetOpSupported(opInst))
    return failure();

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  bool isTargetDevice = ompBuilder->Config.isTargetDevice();
  auto parentFn = opInst.getParentOfType<LLVM::LLVMFuncOp>();
  auto targetOp = cast<omp::TargetOp>(opInst);
  auto &targetRegion = targetOp.getRegion();
  DataLayout dl = DataLayout(opInst.getParentOfType<ModuleOp>());
  SmallVector<Value> mapVars = targetOp.getMapVars();
  llvm::Function *llvmOutlinedFn = nullptr;

  // TODO: It can also be false if a compile-time constant `false` IF clause is
  // specified.
  bool isOffloadEntry =
      isTargetDevice || !ompBuilder->Config.TargetTriples.empty();

  LogicalResult bodyGenStatus = success();
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto bodyCB = [&](InsertPointTy allocaIP,
                    InsertPointTy codeGenIP) -> InsertPointTy {
    // Forward target-cpu and target-features function attributes from the
    // original function to the new outlined function.
    llvm::Function *llvmParentFn =
        moduleTranslation.lookupFunction(parentFn.getName());
    llvmOutlinedFn = codeGenIP.getBlock()->getParent();
    assert(llvmParentFn && llvmOutlinedFn &&
           "Both parent and outlined functions must exist at this point");

    if (auto attr = llvmParentFn->getFnAttribute("target-cpu");
        attr.isStringAttribute())
      llvmOutlinedFn->addFnAttr(attr);

    if (auto attr = llvmParentFn->getFnAttribute("target-features");
        attr.isStringAttribute())
      llvmOutlinedFn->addFnAttr(attr);

    builder.restoreIP(codeGenIP);
    for (auto [argIndex, mapOp] : llvm::enumerate(mapVars)) {
      auto mapInfoOp = cast<omp::MapInfoOp>(mapOp.getDefiningOp());
      llvm::Value *mapOpValue =
          moduleTranslation.lookupValue(mapInfoOp.getVarPtr());
      const auto &arg = targetRegion.front().getArgument(argIndex);
      moduleTranslation.mapValue(arg, mapOpValue);
    }
    llvm::BasicBlock *exitBlock = convertOmpOpRegions(
        targetRegion, "omp.target", builder, moduleTranslation, bodyGenStatus);
    builder.SetInsertPoint(exitBlock);
    return builder.saveIP();
  };

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  StringRef parentName = parentFn.getName();

  llvm::TargetRegionEntryInfo entryInfo;

  if (!getTargetEntryUniqueInfo(entryInfo, targetOp, parentName))
    return failure();

  int32_t defaultValTeams = -1;
  int32_t defaultValThreads = 0;

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  MapInfoData mapData;
  collectMapDataFromMapOperands(mapData, mapVars, moduleTranslation, dl,
                                builder);

  llvm::OpenMPIRBuilder::MapInfosTy combinedInfos;
  auto genMapInfoCB = [&](llvm::OpenMPIRBuilder::InsertPointTy codeGenIP)
      -> llvm::OpenMPIRBuilder::MapInfosTy & {
    builder.restoreIP(codeGenIP);
    genMapInfos(builder, moduleTranslation, dl, combinedInfos, mapData, true);
    return combinedInfos;
  };

  auto argAccessorCB = [&](llvm::Argument &arg, llvm::Value *input,
                           llvm::Value *&retVal, InsertPointTy allocaIP,
                           InsertPointTy codeGenIP) {
    // We just return the unaltered argument for the host function
    // for now, some alterations may be required in the future to
    // keep host fallback functions working identically to the device
    // version (e.g. pass ByCopy values should be treated as such on
    // host and device, currently not always the case)
    if (!isTargetDevice) {
      retVal = cast<llvm::Value>(&arg);
      return codeGenIP;
    }

    return createDeviceArgumentAccessor(mapData, arg, input, retVal, builder,
                                        *ompBuilder, moduleTranslation,
                                        allocaIP, codeGenIP);
  };

  llvm::SmallVector<llvm::Value *, 4> kernelInput;
  for (size_t i = 0; i < mapVars.size(); ++i) {
    // declare target arguments are not passed to kernels as arguments
    // TODO: We currently do not handle cases where a member is explicitly
    // passed in as an argument, this will likley need to be handled in
    // the near future, rather than using IsAMember, it may be better to
    // test if the relevant BlockArg is used within the target region and
    // then use that as a basis for exclusion in the kernel inputs.
    if (!mapData.IsDeclareTarget[i] && !mapData.IsAMember[i])
      kernelInput.push_back(mapData.OriginalValue[i]);
  }
  SmallVector<llvm::OpenMPIRBuilder::DependData> dds;
  buildDependData(targetOp.getDependKinds(), targetOp.getDependVars(),
                  moduleTranslation, dds);

  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createTarget(
      ompLoc, isOffloadEntry, allocaIP, builder.saveIP(), entryInfo,
      defaultValTeams, defaultValThreads, kernelInput, genMapInfoCB, bodyCB,
      argAccessorCB, dds));

  // Remap access operations to declare target reference pointers for the
  // device, essentially generating extra loadop's as necessary
  if (moduleTranslation.getOpenMPBuilder()->Config.isTargetDevice())
    handleDeclareTargetMapVar(mapData, moduleTranslation, builder,
                              llvmOutlinedFn);

  return bodyGenStatus;
}

static LogicalResult
convertDeclareTargetAttr(Operation *op, mlir::omp::DeclareTargetAttr attribute,
                         LLVM::ModuleTranslation &moduleTranslation) {
  // Amend omp.declare_target by deleting the IR of the outlined functions
  // created for target regions. They cannot be filtered out from MLIR earlier
  // because the omp.target operation inside must be translated to LLVM, but
  // the wrapper functions themselves must not remain at the end of the
  // process. We know that functions where omp.declare_target does not match
  // omp.is_target_device at this stage can only be wrapper functions because
  // those that aren't are removed earlier as an MLIR transformation pass.
  if (FunctionOpInterface funcOp = dyn_cast<FunctionOpInterface>(op)) {
    if (auto offloadMod = dyn_cast<omp::OffloadModuleInterface>(
            op->getParentOfType<ModuleOp>().getOperation())) {
      if (!offloadMod.getIsTargetDevice())
        return success();

      omp::DeclareTargetDeviceType declareType =
          attribute.getDeviceType().getValue();

      if (declareType == omp::DeclareTargetDeviceType::host) {
        llvm::Function *llvmFunc =
            moduleTranslation.lookupFunction(funcOp.getName());
        llvmFunc->dropAllReferences();
        llvmFunc->eraseFromParent();
      }
    }
    return success();
  }

  if (LLVM::GlobalOp gOp = dyn_cast<LLVM::GlobalOp>(op)) {
    llvm::Module *llvmModule = moduleTranslation.getLLVMModule();
    if (auto *gVal = llvmModule->getNamedValue(gOp.getSymName())) {
      llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
      bool isDeclaration = gOp.isDeclaration();
      bool isExternallyVisible =
          gOp.getVisibility() != mlir::SymbolTable::Visibility::Private;
      auto loc = op->getLoc()->findInstanceOf<FileLineColLoc>();
      llvm::StringRef mangledName = gOp.getSymName();
      auto captureClause =
          convertToCaptureClauseKind(attribute.getCaptureClause().getValue());
      auto deviceClause =
          convertToDeviceClauseKind(attribute.getDeviceType().getValue());
      // unused for MLIR at the moment, required in Clang for book
      // keeping
      std::vector<llvm::GlobalVariable *> generatedRefs;

      std::vector<llvm::Triple> targetTriple;
      auto targetTripleAttr = dyn_cast_or_null<mlir::StringAttr>(
          op->getParentOfType<mlir::ModuleOp>()->getAttr(
              LLVM::LLVMDialect::getTargetTripleAttrName()));
      if (targetTripleAttr)
        targetTriple.emplace_back(targetTripleAttr.data());

      auto fileInfoCallBack = [&loc]() {
        std::string filename = "";
        std::uint64_t lineNo = 0;

        if (loc) {
          filename = loc.getFilename().str();
          lineNo = loc.getLine();
        }

        return std::pair<std::string, std::uint64_t>(llvm::StringRef(filename),
                                                     lineNo);
      };

      ompBuilder->registerTargetGlobalVariable(
          captureClause, deviceClause, isDeclaration, isExternallyVisible,
          ompBuilder->getTargetEntryUniqueInfo(fileInfoCallBack), mangledName,
          generatedRefs, /*OpenMPSimd*/ false, targetTriple,
          /*GlobalInitializer*/ nullptr, /*VariableLinkage*/ nullptr,
          gVal->getType(), gVal);

      if (ompBuilder->Config.isTargetDevice() &&
          (attribute.getCaptureClause().getValue() !=
               mlir::omp::DeclareTargetCaptureClause::to ||
           ompBuilder->Config.hasRequiresUnifiedSharedMemory())) {
        ompBuilder->getAddrOfDeclareTargetVar(
            captureClause, deviceClause, isDeclaration, isExternallyVisible,
            ompBuilder->getTargetEntryUniqueInfo(fileInfoCallBack), mangledName,
            generatedRefs, /*OpenMPSimd*/ false, targetTriple, gVal->getType(),
            /*GlobalInitializer*/ nullptr,
            /*VariableLinkage*/ nullptr);
      }
    }
  }

  return success();
}

// Returns true if the operation is inside a TargetOp or
// is part of a declare target function.
static bool isTargetDeviceOp(Operation *op) {
  // Assumes no reverse offloading
  if (op->getParentOfType<omp::TargetOp>())
    return true;

  if (auto parentFn = op->getParentOfType<LLVM::LLVMFuncOp>())
    if (auto declareTargetIface =
            llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
                parentFn.getOperation()))
      if (declareTargetIface.isDeclareTarget() &&
          declareTargetIface.getDeclareTargetDeviceType() !=
              mlir::omp::DeclareTargetDeviceType::host)
        return true;

  return false;
}

/// Given an OpenMP MLIR operation, create the corresponding LLVM IR
/// (including OpenMP runtime calls).
static LogicalResult
convertHostOrTargetOperation(Operation *op, llvm::IRBuilderBase &builder,
                             LLVM::ModuleTranslation &moduleTranslation) {

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](omp::BarrierOp) {
        ompBuilder->createBarrier(builder.saveIP(), llvm::omp::OMPD_barrier);
        return success();
      })
      .Case([&](omp::TaskyieldOp) {
        ompBuilder->createTaskyield(builder.saveIP());
        return success();
      })
      .Case([&](omp::FlushOp) {
        // No support in Openmp runtime function (__kmpc_flush) to accept
        // the argument list.
        // OpenMP standard states the following:
        //  "An implementation may implement a flush with a list by ignoring
        //   the list, and treating it the same as a flush without a list."
        //
        // The argument list is discarded so that, flush with a list is treated
        // same as a flush without a list.
        ompBuilder->createFlush(builder.saveIP());
        return success();
      })
      .Case([&](omp::ParallelOp op) {
        return convertOmpParallel(op, builder, moduleTranslation);
      })
      .Case([&](omp::MaskedOp) {
        return convertOmpMasked(*op, builder, moduleTranslation);
      })
      .Case([&](omp::MasterOp) {
        return convertOmpMaster(*op, builder, moduleTranslation);
      })
      .Case([&](omp::CriticalOp) {
        return convertOmpCritical(*op, builder, moduleTranslation);
      })
      .Case([&](omp::OrderedRegionOp) {
        return convertOmpOrderedRegion(*op, builder, moduleTranslation);
      })
      .Case([&](omp::OrderedOp) {
        return convertOmpOrdered(*op, builder, moduleTranslation);
      })
      .Case([&](omp::WsloopOp) {
        return convertOmpWsloop(*op, builder, moduleTranslation);
      })
      .Case([&](omp::SimdOp) {
        return convertOmpSimd(*op, builder, moduleTranslation);
      })
      .Case([&](omp::AtomicReadOp) {
        return convertOmpAtomicRead(*op, builder, moduleTranslation);
      })
      .Case([&](omp::AtomicWriteOp) {
        return convertOmpAtomicWrite(*op, builder, moduleTranslation);
      })
      .Case([&](omp::AtomicUpdateOp op) {
        return convertOmpAtomicUpdate(op, builder, moduleTranslation);
      })
      .Case([&](omp::AtomicCaptureOp op) {
        return convertOmpAtomicCapture(op, builder, moduleTranslation);
      })
      .Case([&](omp::SectionsOp) {
        return convertOmpSections(*op, builder, moduleTranslation);
      })
      .Case([&](omp::SingleOp op) {
        return convertOmpSingle(op, builder, moduleTranslation);
      })
      .Case([&](omp::TeamsOp op) {
        return convertOmpTeams(op, builder, moduleTranslation);
      })
      .Case([&](omp::TaskOp op) {
        return convertOmpTaskOp(op, builder, moduleTranslation);
      })
      .Case([&](omp::TaskgroupOp op) {
        return convertOmpTaskgroupOp(op, builder, moduleTranslation);
      })
      .Case([&](omp::TaskwaitOp op) {
        return convertOmpTaskwaitOp(op, builder, moduleTranslation);
      })
      .Case<omp::YieldOp, omp::TerminatorOp, omp::DeclareReductionOp,
            omp::CriticalDeclareOp>([](auto op) {
        // `yield` and `terminator` can be just omitted. The block structure
        // was created in the region that handles their parent operation.
        // `declare_reduction` will be used by reductions and is not
        // converted directly, skip it.
        // `critical.declare` is only used to declare names of critical
        // sections which will be used by `critical` ops and hence can be
        // ignored for lowering. The OpenMP IRBuilder will create unique
        // name for critical section names.
        return success();
      })
      .Case([&](omp::ThreadprivateOp) {
        return convertOmpThreadprivate(*op, builder, moduleTranslation);
      })
      .Case<omp::TargetDataOp, omp::TargetEnterDataOp, omp::TargetExitDataOp,
            omp::TargetUpdateOp>([&](auto op) {
        return convertOmpTargetData(op, builder, moduleTranslation);
      })
      .Case([&](omp::TargetOp) {
        return convertOmpTarget(*op, builder, moduleTranslation);
      })
      .Case<omp::MapInfoOp, omp::MapBoundsOp, omp::PrivateClauseOp>(
          [&](auto op) {
            // No-op, should be handled by relevant owning operations e.g.
            // TargetOp, TargetEnterDataOp, TargetExitDataOp, TargetDataOp etc.
            // and then discarded
            return success();
          })
      .Default([&](Operation *inst) {
        return inst->emitError("unsupported OpenMP operation: ")
               << inst->getName();
      });
}

static LogicalResult
convertTargetDeviceOp(Operation *op, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  return convertHostOrTargetOperation(op, builder, moduleTranslation);
}

static LogicalResult
convertTargetOpsInNest(Operation *op, llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  if (isa<omp::TargetOp>(op))
    return convertOmpTarget(*op, builder, moduleTranslation);
  if (isa<omp::TargetDataOp>(op))
    return convertOmpTargetData(op, builder, moduleTranslation);
  bool interrupted =
      op->walk<WalkOrder::PreOrder>([&](Operation *oper) {
          if (isa<omp::TargetOp>(oper)) {
            if (failed(convertOmpTarget(*oper, builder, moduleTranslation)))
              return WalkResult::interrupt();
            return WalkResult::skip();
          }
          if (isa<omp::TargetDataOp>(oper)) {
            if (failed(convertOmpTargetData(oper, builder, moduleTranslation)))
              return WalkResult::interrupt();
            return WalkResult::skip();
          }
          return WalkResult::advance();
        }).wasInterrupted();
  return failure(interrupted);
}

namespace {

/// Implementation of the dialect interface that converts operations belonging
/// to the OpenMP dialect to LLVM IR.
class OpenMPDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final;

  /// Given an OpenMP MLIR attribute, create the corresponding LLVM-IR,
  /// runtime calls, or operation amendments
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final;
};

} // namespace

LogicalResult OpenMPDialectLLVMIRTranslationInterface::amendOperation(
    Operation *op, ArrayRef<llvm::Instruction *> instructions,
    NamedAttribute attribute,
    LLVM::ModuleTranslation &moduleTranslation) const {
  return llvm::StringSwitch<llvm::function_ref<LogicalResult(Attribute)>>(
             attribute.getName())
      .Case("omp.is_target_device",
            [&](Attribute attr) {
              if (auto deviceAttr = dyn_cast<BoolAttr>(attr)) {
                llvm::OpenMPIRBuilderConfig &config =
                    moduleTranslation.getOpenMPBuilder()->Config;
                config.setIsTargetDevice(deviceAttr.getValue());
                return success();
              }
              return failure();
            })
      .Case("omp.is_gpu",
            [&](Attribute attr) {
              if (auto gpuAttr = dyn_cast<BoolAttr>(attr)) {
                llvm::OpenMPIRBuilderConfig &config =
                    moduleTranslation.getOpenMPBuilder()->Config;
                config.setIsGPU(gpuAttr.getValue());
                return success();
              }
              return failure();
            })
      .Case("omp.host_ir_filepath",
            [&](Attribute attr) {
              if (auto filepathAttr = dyn_cast<StringAttr>(attr)) {
                llvm::OpenMPIRBuilder *ompBuilder =
                    moduleTranslation.getOpenMPBuilder();
                ompBuilder->loadOffloadInfoMetadata(filepathAttr.getValue());
                return success();
              }
              return failure();
            })
      .Case("omp.flags",
            [&](Attribute attr) {
              if (auto rtlAttr = dyn_cast<omp::FlagsAttr>(attr))
                return convertFlagsAttr(op, rtlAttr, moduleTranslation);
              return failure();
            })
      .Case("omp.version",
            [&](Attribute attr) {
              if (auto versionAttr = dyn_cast<omp::VersionAttr>(attr)) {
                llvm::OpenMPIRBuilder *ompBuilder =
                    moduleTranslation.getOpenMPBuilder();
                ompBuilder->M.addModuleFlag(llvm::Module::Max, "openmp",
                                            versionAttr.getVersion());
                return success();
              }
              return failure();
            })
      .Case("omp.declare_target",
            [&](Attribute attr) {
              if (auto declareTargetAttr =
                      dyn_cast<omp::DeclareTargetAttr>(attr))
                return convertDeclareTargetAttr(op, declareTargetAttr,
                                                moduleTranslation);
              return failure();
            })
      .Case("omp.requires",
            [&](Attribute attr) {
              if (auto requiresAttr = dyn_cast<omp::ClauseRequiresAttr>(attr)) {
                using Requires = omp::ClauseRequires;
                Requires flags = requiresAttr.getValue();
                llvm::OpenMPIRBuilderConfig &config =
                    moduleTranslation.getOpenMPBuilder()->Config;
                config.setHasRequiresReverseOffload(
                    bitEnumContainsAll(flags, Requires::reverse_offload));
                config.setHasRequiresUnifiedAddress(
                    bitEnumContainsAll(flags, Requires::unified_address));
                config.setHasRequiresUnifiedSharedMemory(
                    bitEnumContainsAll(flags, Requires::unified_shared_memory));
                config.setHasRequiresDynamicAllocators(
                    bitEnumContainsAll(flags, Requires::dynamic_allocators));
                return success();
              }
              return failure();
            })
      .Case("omp.target_triples",
            [&](Attribute attr) {
              if (auto triplesAttr = dyn_cast<ArrayAttr>(attr)) {
                llvm::OpenMPIRBuilderConfig &config =
                    moduleTranslation.getOpenMPBuilder()->Config;
                config.TargetTriples.clear();
                config.TargetTriples.reserve(triplesAttr.size());
                for (Attribute tripleAttr : triplesAttr) {
                  if (auto tripleStrAttr = dyn_cast<StringAttr>(tripleAttr))
                    config.TargetTriples.emplace_back(tripleStrAttr.getValue());
                  else
                    return failure();
                }
                return success();
              }
              return failure();
            })
      .Default([](Attribute) {
        // Fall through for omp attributes that do not require lowering.
        return success();
      })(attribute.getValue());

  return failure();
}

/// Given an OpenMP MLIR operation, create the corresponding LLVM IR
/// (including OpenMP runtime calls).
LogicalResult OpenMPDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  if (ompBuilder->Config.isTargetDevice()) {
    if (isTargetDeviceOp(op)) {
      return convertTargetDeviceOp(op, builder, moduleTranslation);
    } else {
      return convertTargetOpsInNest(op, builder, moduleTranslation);
    }
  }
  return convertHostOrTargetOperation(op, builder, moduleTranslation);
}

void mlir::registerOpenMPDialectTranslation(DialectRegistry &registry) {
  registry.insert<omp::OpenMPDialect>();
  registry.addExtension(+[](MLIRContext *ctx, omp::OpenMPDialect *dialect) {
    dialect->addInterfaces<OpenMPDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerOpenMPDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerOpenMPDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
