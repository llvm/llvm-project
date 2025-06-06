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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
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

/// Stack frame to hold a \see llvm::CanonicalLoopInfo representing the
/// collapsed canonical loop information corresponding to an \c omp.loop_nest
/// operation.
class OpenMPLoopInfoStackFrame
    : public LLVM::ModuleTranslation::StackFrameBase<OpenMPLoopInfoStackFrame> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpenMPLoopInfoStackFrame)
  llvm::CanonicalLoopInfo *loopInfo = nullptr;
};

/// Custom error class to signal translation errors that don't need reporting,
/// since encountering them will have already triggered relevant error messages.
///
/// Its purpose is to serve as the glue between MLIR failures represented as
/// \see LogicalResult instances and \see llvm::Error instances used to
/// propagate errors through the \see llvm::OpenMPIRBuilder. Generally, when an
/// error of the first type is raised, a message is emitted directly (the \see
/// LogicalResult itself does not hold any information). If we need to forward
/// this error condition as an \see llvm::Error while avoiding triggering some
/// redundant error reporting later on, we need a custom \see llvm::ErrorInfo
/// class to just signal this situation has happened.
///
/// For example, this class should be used to trigger errors from within
/// callbacks passed to the \see OpenMPIRBuilder when they were triggered by the
/// translation of their own regions. This unclutters the error log from
/// redundant messages.
class PreviouslyReportedError
    : public llvm::ErrorInfo<PreviouslyReportedError> {
public:
  void log(raw_ostream &) const override {
    // Do not log anything.
  }

  std::error_code convertToErrorCode() const override {
    llvm_unreachable(
        "PreviouslyReportedError doesn't support ECError conversion");
  }

  // Used by ErrorInfo::classID.
  static char ID;
};

char PreviouslyReportedError::ID = 0;

/*
 * Custom class for processing linear clause for omp.wsloop
 * and omp.simd. Linear clause translation requires setup,
 * initialization, update, and finalization at varying
 * basic blocks in the IR. This class helps maintain
 * internal state to allow consistent translation in
 * each of these stages.
 */

class LinearClauseProcessor {

private:
  SmallVector<llvm::Value *> linearPreconditionVars;
  SmallVector<llvm::Value *> linearLoopBodyTemps;
  SmallVector<llvm::AllocaInst *> linearOrigVars;
  SmallVector<llvm::Value *> linearOrigVal;
  SmallVector<llvm::Value *> linearSteps;
  llvm::BasicBlock *linearFinalizationBB;
  llvm::BasicBlock *linearExitBB;
  llvm::BasicBlock *linearLastIterExitBB;

public:
  // Allocate space for linear variabes
  void createLinearVar(llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation,
                       mlir::Value &linearVar) {
    if (llvm::AllocaInst *linearVarAlloca = dyn_cast<llvm::AllocaInst>(
            moduleTranslation.lookupValue(linearVar))) {
      linearPreconditionVars.push_back(builder.CreateAlloca(
          linearVarAlloca->getAllocatedType(), nullptr, ".linear_var"));
      llvm::Value *linearLoopBodyTemp = builder.CreateAlloca(
          linearVarAlloca->getAllocatedType(), nullptr, ".linear_result");
      linearOrigVal.push_back(moduleTranslation.lookupValue(linearVar));
      linearLoopBodyTemps.push_back(linearLoopBodyTemp);
      linearOrigVars.push_back(linearVarAlloca);
    }
  }

  // Initialize linear step
  inline void initLinearStep(LLVM::ModuleTranslation &moduleTranslation,
                             mlir::Value &linearStep) {
    linearSteps.push_back(moduleTranslation.lookupValue(linearStep));
  }

  // Emit IR for initialization of linear variables
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy
  initLinearVar(llvm::IRBuilderBase &builder,
                LLVM::ModuleTranslation &moduleTranslation,
                llvm::BasicBlock *loopPreHeader) {
    builder.SetInsertPoint(loopPreHeader->getTerminator());
    for (size_t index = 0; index < linearOrigVars.size(); index++) {
      llvm::LoadInst *linearVarLoad = builder.CreateLoad(
          linearOrigVars[index]->getAllocatedType(), linearOrigVars[index]);
      builder.CreateStore(linearVarLoad, linearPreconditionVars[index]);
    }
    llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterBarrierIP =
        moduleTranslation.getOpenMPBuilder()->createBarrier(
            builder.saveIP(), llvm::omp::OMPD_barrier);
    return afterBarrierIP;
  }

  // Emit IR for updating Linear variables
  void updateLinearVar(llvm::IRBuilderBase &builder, llvm::BasicBlock *loopBody,
                       llvm::Value *loopInductionVar) {
    builder.SetInsertPoint(loopBody->getTerminator());
    for (size_t index = 0; index < linearPreconditionVars.size(); index++) {
      // Emit increments for linear vars
      llvm::LoadInst *linearVarStart =
          builder.CreateLoad(linearOrigVars[index]->getAllocatedType(),

                             linearPreconditionVars[index]);
      auto mulInst = builder.CreateMul(loopInductionVar, linearSteps[index]);
      auto addInst = builder.CreateAdd(linearVarStart, mulInst);
      builder.CreateStore(addInst, linearLoopBodyTemps[index]);
    }
  }

  // Linear variable finalization is conditional on the last logical iteration.
  // Create BB splits to manage the same.
  void outlineLinearFinalizationBB(llvm::IRBuilderBase &builder,
                                   llvm::BasicBlock *loopExit) {
    linearFinalizationBB = loopExit->splitBasicBlock(
        loopExit->getTerminator(), "omp_loop.linear_finalization");
    linearExitBB = linearFinalizationBB->splitBasicBlock(
        linearFinalizationBB->getTerminator(), "omp_loop.linear_exit");
    linearLastIterExitBB = linearFinalizationBB->splitBasicBlock(
        linearFinalizationBB->getTerminator(), "omp_loop.linear_lastiter_exit");
  }

  // Finalize the linear vars
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy
  finalizeLinearVar(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    llvm::Value *lastIter) {
    // Emit condition to check whether last logical iteration is being executed
    builder.SetInsertPoint(linearFinalizationBB->getTerminator());
    llvm::Value *loopLastIterLoad = builder.CreateLoad(
        llvm::Type::getInt32Ty(builder.getContext()), lastIter);
    llvm::Value *isLast =
        builder.CreateCmp(llvm::CmpInst::ICMP_NE, loopLastIterLoad,
                          llvm::ConstantInt::get(
                              llvm::Type::getInt32Ty(builder.getContext()), 0));
    // Store the linear variable values to original variables.
    builder.SetInsertPoint(linearLastIterExitBB->getTerminator());
    for (size_t index = 0; index < linearOrigVars.size(); index++) {
      llvm::LoadInst *linearVarTemp =
          builder.CreateLoad(linearOrigVars[index]->getAllocatedType(),
                             linearLoopBodyTemps[index]);
      builder.CreateStore(linearVarTemp, linearOrigVars[index]);
    }

    // Create conditional branch such that the linear variable
    // values are stored to original variables only at the
    // last logical iteration
    builder.SetInsertPoint(linearFinalizationBB->getTerminator());
    builder.CreateCondBr(isLast, linearLastIterExitBB, linearExitBB);
    linearFinalizationBB->getTerminator()->eraseFromParent();
    // Emit barrier
    builder.SetInsertPoint(linearExitBB->getTerminator());
    return moduleTranslation.getOpenMPBuilder()->createBarrier(
        builder.saveIP(), llvm::omp::OMPD_barrier);
  }

  // Rewrite all uses of the original variable in `BBName`
  //  with the linear variable in-place
  void rewriteInPlace(llvm::IRBuilderBase &builder, std::string BBName,
                      size_t varIndex) {
    llvm::SmallVector<llvm::User *> users;
    for (llvm::User *user : linearOrigVal[varIndex]->users())
      users.push_back(user);
    for (auto *user : users) {
      if (auto *userInst = dyn_cast<llvm::Instruction>(user)) {
        if (userInst->getParent()->getName().str() == BBName)
          user->replaceUsesOfWith(linearOrigVal[varIndex],
                                  linearLoopBodyTemps[varIndex]);
      }
    }
  }
};

} // namespace

/// Looks up from the operation from and returns the PrivateClauseOp with
/// name symbolName
static omp::PrivateClauseOp findPrivatizer(Operation *from,
                                           SymbolRefAttr symbolName) {
  omp::PrivateClauseOp privatizer =
      SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(from,
                                                                 symbolName);
  assert(privatizer && "privatizer not found in the symbol table");
  return privatizer;
}

/// Check whether translation to LLVM IR for the given operation is currently
/// supported. If not, descriptive diagnostics will be emitted to let users know
/// this is a not-yet-implemented feature.
///
/// \returns success if no unimplemented features are needed to translate the
///          given operation.
static LogicalResult checkImplementationStatus(Operation &op) {
  auto todo = [&op](StringRef clauseName) {
    return op.emitError() << "not yet implemented: Unhandled clause "
                          << clauseName << " in " << op.getName()
                          << " operation";
  };

  auto checkAllocate = [&todo](auto op, LogicalResult &result) {
    if (!op.getAllocateVars().empty() || !op.getAllocatorVars().empty())
      result = todo("allocate");
  };
  auto checkBare = [&todo](auto op, LogicalResult &result) {
    if (op.getBare())
      result = todo("ompx_bare");
  };
  auto checkCancelDirective = [&todo](auto op, LogicalResult &result) {
    omp::ClauseCancellationConstructType cancelledDirective =
        op.getCancelDirective();
    // Cancelling a taskloop is not yet supported because we don't yet have LLVM
    // IR conversion for taskloop
    if (cancelledDirective == omp::ClauseCancellationConstructType::Taskgroup) {
      Operation *parent = op->getParentOp();
      while (parent) {
        if (parent->getDialect() == op->getDialect())
          break;
        parent = parent->getParentOp();
      }
      if (isa_and_nonnull<omp::TaskloopOp>(parent))
        result = todo("cancel directive inside of taskloop");
    }
  };
  auto checkDepend = [&todo](auto op, LogicalResult &result) {
    if (!op.getDependVars().empty() || op.getDependKinds())
      result = todo("depend");
  };
  auto checkDevice = [&todo](auto op, LogicalResult &result) {
    if (op.getDevice())
      result = todo("device");
  };
  auto checkDistSchedule = [&todo](auto op, LogicalResult &result) {
    if (op.getDistScheduleChunkSize())
      result = todo("dist_schedule with chunk_size");
  };
  auto checkHint = [](auto op, LogicalResult &) {
    if (op.getHint())
      op.emitWarning("hint clause discarded");
  };
  auto checkInReduction = [&todo](auto op, LogicalResult &result) {
    if (!op.getInReductionVars().empty() || op.getInReductionByref() ||
        op.getInReductionSyms())
      result = todo("in_reduction");
  };
  auto checkIsDevicePtr = [&todo](auto op, LogicalResult &result) {
    if (!op.getIsDevicePtrVars().empty())
      result = todo("is_device_ptr");
  };
  auto checkLinear = [&todo](auto op, LogicalResult &result) {
    if (!op.getLinearVars().empty() || !op.getLinearStepVars().empty())
      result = todo("linear");
  };
  auto checkNowait = [&todo](auto op, LogicalResult &result) {
    if (op.getNowait())
      result = todo("nowait");
  };
  auto checkOrder = [&todo](auto op, LogicalResult &result) {
    if (op.getOrder() || op.getOrderMod())
      result = todo("order");
  };
  auto checkParLevelSimd = [&todo](auto op, LogicalResult &result) {
    if (op.getParLevelSimd())
      result = todo("parallelization-level");
  };
  auto checkPriority = [&todo](auto op, LogicalResult &result) {
    if (op.getPriority())
      result = todo("priority");
  };
  auto checkPrivate = [&todo](auto op, LogicalResult &result) {
    if constexpr (std::is_same_v<std::decay_t<decltype(op)>, omp::TargetOp>) {
      // Privatization is supported only for included target tasks.
      if (!op.getPrivateVars().empty() && op.getNowait())
        result = todo("privatization for deferred target tasks");
    } else {
      if (!op.getPrivateVars().empty() || op.getPrivateSyms())
        result = todo("privatization");
    }
  };
  auto checkReduction = [&todo](auto op, LogicalResult &result) {
    if (isa<omp::TeamsOp>(op) || isa<omp::SimdOp>(op))
      if (!op.getReductionVars().empty() || op.getReductionByref() ||
          op.getReductionSyms())
        result = todo("reduction");
    if (op.getReductionMod() &&
        op.getReductionMod().value() != omp::ReductionModifier::defaultmod)
      result = todo("reduction with modifier");
  };
  auto checkTaskReduction = [&todo](auto op, LogicalResult &result) {
    if (!op.getTaskReductionVars().empty() || op.getTaskReductionByref() ||
        op.getTaskReductionSyms())
      result = todo("task_reduction");
  };
  auto checkUntied = [&todo](auto op, LogicalResult &result) {
    if (op.getUntied())
      result = todo("untied");
  };

  LogicalResult result = success();
  llvm::TypeSwitch<Operation &>(op)
      .Case([&](omp::CancelOp op) { checkCancelDirective(op, result); })
      .Case([&](omp::CancellationPointOp op) {
        checkCancelDirective(op, result);
      })
      .Case([&](omp::DistributeOp op) {
        checkAllocate(op, result);
        checkDistSchedule(op, result);
        checkOrder(op, result);
      })
      .Case([&](omp::OrderedRegionOp op) { checkParLevelSimd(op, result); })
      .Case([&](omp::SectionsOp op) {
        checkAllocate(op, result);
        checkPrivate(op, result);
        checkReduction(op, result);
      })
      .Case([&](omp::SingleOp op) {
        checkAllocate(op, result);
        checkPrivate(op, result);
      })
      .Case([&](omp::TeamsOp op) {
        checkAllocate(op, result);
        checkPrivate(op, result);
      })
      .Case([&](omp::TaskOp op) {
        checkAllocate(op, result);
        checkInReduction(op, result);
      })
      .Case([&](omp::TaskgroupOp op) {
        checkAllocate(op, result);
        checkTaskReduction(op, result);
      })
      .Case([&](omp::TaskwaitOp op) {
        checkDepend(op, result);
        checkNowait(op, result);
      })
      .Case([&](omp::TaskloopOp op) {
        // TODO: Add other clauses check
        checkUntied(op, result);
        checkPriority(op, result);
      })
      .Case([&](omp::WsloopOp op) {
        checkAllocate(op, result);
        checkOrder(op, result);
        checkReduction(op, result);
      })
      .Case([&](omp::ParallelOp op) {
        checkAllocate(op, result);
        checkReduction(op, result);
      })
      .Case([&](omp::SimdOp op) {
        checkLinear(op, result);
        checkReduction(op, result);
      })
      .Case<omp::AtomicReadOp, omp::AtomicWriteOp, omp::AtomicUpdateOp,
            omp::AtomicCaptureOp>([&](auto op) { checkHint(op, result); })
      .Case<omp::TargetEnterDataOp, omp::TargetExitDataOp, omp::TargetUpdateOp>(
          [&](auto op) { checkDepend(op, result); })
      .Case([&](omp::TargetOp op) {
        checkAllocate(op, result);
        checkBare(op, result);
        checkDevice(op, result);
        checkInReduction(op, result);
        checkIsDevicePtr(op, result);
        checkPrivate(op, result);
      })
      .Default([](Operation &) {
        // Assume all clauses for an operation can be translated unless they are
        // checked above.
      });
  return result;
}

static LogicalResult handleError(llvm::Error error, Operation &op) {
  LogicalResult result = success();
  if (error) {
    llvm::handleAllErrors(
        std::move(error),
        [&](const PreviouslyReportedError &) { result = failure(); },
        [&](const llvm::ErrorInfoBase &err) {
          result = op.emitError(err.message());
        });
  }
  return result;
}

template <typename T>
static LogicalResult handleError(llvm::Expected<T> &result, Operation &op) {
  if (!result)
    return handleError(result.takeError(), op);

  return success();
}

/// Find the insertion point for allocas given the current insertion point for
/// normal operations in the builder.
static llvm::OpenMPIRBuilder::InsertPointTy
findAllocaInsertPoint(llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  // If there is an alloca insertion point on stack, i.e. we are in a nested
  // operation and a specific point was provided by some surrounding operation,
  // use it.
  llvm::OpenMPIRBuilder::InsertPointTy allocaInsertPoint;
  WalkResult walkResult = moduleTranslation.stackWalk<OpenMPAllocaStackFrame>(
      [&](OpenMPAllocaStackFrame &frame) {
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

/// Find the loop information structure for the loop nest being translated. It
/// will return a `null` value unless called from the translation function for
/// a loop wrapper operation after successfully translating its body.
static llvm::CanonicalLoopInfo *
findCurrentLoopInfo(LLVM::ModuleTranslation &moduleTranslation) {
  llvm::CanonicalLoopInfo *loopInfo = nullptr;
  moduleTranslation.stackWalk<OpenMPLoopInfoStackFrame>(
      [&](OpenMPLoopInfoStackFrame &frame) {
        loopInfo = frame.loopInfo;
        return WalkResult::interrupt();
      });
  return loopInfo;
}

/// Converts the given region that appears within an OpenMP dialect operation to
/// LLVM IR, creating a branch from the `sourceBlock` to the entry block of the
/// region, and a branch from any block with an successor-less OpenMP terminator
/// to `continuationBlock`. Populates `continuationBlockPHIs` with the PHI nodes
/// of the continuation block if provided.
static llvm::Expected<llvm::BasicBlock *> convertOmpOpRegions(
    Region &region, StringRef blockName, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation,
    SmallVectorImpl<llvm::PHINode *> *continuationBlockPHIs = nullptr) {
  bool isLoopWrapper = isa<omp::LoopWrapperInterface>(region.getParentOp());

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
  // operands in preparation of creating PHI nodes. This is skipped for loop
  // wrapper operations, for which we know in advance they have no terminators.
  SmallVector<llvm::Type *> continuationBlockPHITypes;
  unsigned numYields = 0;

  if (!isLoopWrapper) {
    bool operandsProcessed = false;
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
            moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder)))
      return llvm::make_error<PreviouslyReportedError>();

    // Create a direct branch here for loop wrappers to prevent their lack of a
    // terminator from causing a crash below.
    if (isLoopWrapper) {
      builder.CreateBr(continuationBlock);
      continue;
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

/// Maps block arguments from \p blockArgIface (which are MLIR values) to the
/// corresponding LLVM values of \p the interface's operands. This is useful
/// when an OpenMP region with entry block arguments is converted to LLVM. In
/// this case the block arguments are (part of) of the OpenMP region's entry
/// arguments and the operands are (part of) of the operands to the OpenMP op
/// containing the region.
static void forwardArgs(LLVM::ModuleTranslation &moduleTranslation,
                        omp::BlockArgOpenMPOpInterface blockArgIface) {
  llvm::SmallVector<std::pair<Value, BlockArgument>> blockArgsPairs;
  blockArgIface.getBlockArgsPairs(blockArgsPairs);
  for (auto [var, arg] : blockArgsPairs)
    moduleTranslation.mapValue(arg, moduleTranslation.lookupValue(var));
}

/// Helper function to map block arguments defined by ignored loop wrappers to
/// LLVM values and prevent any uses of those from triggering null pointer
/// dereferences.
///
/// This must be called after block arguments of parent wrappers have already
/// been mapped to LLVM IR values.
static LogicalResult
convertIgnoredWrapper(omp::LoopWrapperInterface opInst,
                      LLVM::ModuleTranslation &moduleTranslation) {
  // Map block arguments directly to the LLVM value associated to the
  // corresponding operand. This is semantically equivalent to this wrapper not
  // being present.
  return llvm::TypeSwitch<Operation *, LogicalResult>(opInst)
      .Case([&](omp::SimdOp op) {
        forwardArgs(moduleTranslation,
                    cast<omp::BlockArgOpenMPOpInterface>(*op));
        op.emitWarning() << "simd information on composite construct discarded";
        return success();
      })
      .Default([&](Operation *op) {
        return op->emitError() << "cannot ignore wrapper";
      });
}

/// Converts an OpenMP 'masked' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpMasked(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  auto maskedOp = cast<omp::MaskedOp>(opInst);
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;

  if (failed(checkImplementationStatus(opInst)))
    return failure();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // MaskedOp has only one region associated with it.
    auto &region = maskedOp.getRegion();
    builder.restoreIP(codeGenIP);
    return convertOmpOpRegions(region, "omp.masked.region", builder,
                               moduleTranslation)
        .takeError();
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) { return llvm::Error::success(); };

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
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createMasked(ompLoc, bodyGenCB,
                                                         finiCB, filterVal);

  if (failed(handleError(afterIP, opInst)))
    return failure();

  builder.restoreIP(*afterIP);
  return success();
}

/// Converts an OpenMP 'master' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpMaster(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto masterOp = cast<omp::MasterOp>(opInst);

  if (failed(checkImplementationStatus(opInst)))
    return failure();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // MasterOp has only one region associated with it.
    auto &region = masterOp.getRegion();
    builder.restoreIP(codeGenIP);
    return convertOmpOpRegions(region, "omp.master.region", builder,
                               moduleTranslation)
        .takeError();
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) { return llvm::Error::success(); };

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createMaster(ompLoc, bodyGenCB,
                                                         finiCB);

  if (failed(handleError(afterIP, opInst)))
    return failure();

  builder.restoreIP(*afterIP);
  return success();
}

/// Converts an OpenMP 'critical' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpCritical(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto criticalOp = cast<omp::CriticalOp>(opInst);

  if (failed(checkImplementationStatus(opInst)))
    return failure();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // CriticalOp has only one region associated with it.
    auto &region = cast<omp::CriticalOp>(opInst).getRegion();
    builder.restoreIP(codeGenIP);
    return convertOmpOpRegions(region, "omp.critical.region", builder,
                               moduleTranslation)
        .takeError();
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) { return llvm::Error::success(); };

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
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createCritical(
          ompLoc, bodyGenCB, finiCB, criticalOp.getName().value_or(""), hint);

  if (failed(handleError(afterIP, opInst)))
    return failure();

  builder.restoreIP(*afterIP);
  return success();
}

/// A util to collect info needed to convert delayed privatizers from MLIR to
/// LLVM.
struct PrivateVarsInfo {
  template <typename OP>
  PrivateVarsInfo(OP op)
      : blockArgs(
            cast<omp::BlockArgOpenMPOpInterface>(*op).getPrivateBlockArgs()) {
    mlirVars.reserve(blockArgs.size());
    llvmVars.reserve(blockArgs.size());
    collectPrivatizationDecls<OP>(op);

    for (mlir::Value privateVar : op.getPrivateVars())
      mlirVars.push_back(privateVar);
  }

  MutableArrayRef<BlockArgument> blockArgs;
  SmallVector<mlir::Value> mlirVars;
  SmallVector<llvm::Value *> llvmVars;
  SmallVector<omp::PrivateClauseOp> privatizers;

private:
  /// Populates `privatizations` with privatization declarations used for the
  /// given op.
  template <class OP>
  void collectPrivatizationDecls(OP op) {
    std::optional<ArrayAttr> attr = op.getPrivateSyms();
    if (!attr)
      return;

    privatizers.reserve(privatizers.size() + attr->size());
    for (auto symbolRef : attr->getAsRange<SymbolRefAttr>()) {
      privatizers.push_back(findPrivatizer(op, symbolRef));
    }
  }
};

/// Populates `reductions` with reduction declarations used in the given op.
template <typename T>
static void
collectReductionDecls(T op,
                      SmallVectorImpl<omp::DeclareReductionOp> &reductions) {
  std::optional<ArrayAttr> attr = op.getReductionSyms();
  if (!attr)
    return;

  reductions.reserve(reductions.size() + op.getNumReductionVars());
  for (auto symbolRef : attr->getAsRange<SymbolRefAttr>()) {
    reductions.push_back(
        SymbolTable::lookupNearestSymbolFrom<omp::DeclareReductionOp>(
            op, symbolRef));
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

  SmallVector<llvm::PHINode *> phis;
  llvm::Expected<llvm::BasicBlock *> continuationBlock =
      convertOmpOpRegions(region, blockName, builder, moduleTranslation, &phis);

  if (failed(handleError(continuationBlock, *region.getParentOp())))
    return failure();

  if (continuationBlockArgs)
    llvm::append_range(*continuationBlockArgs, phis);
  builder.SetInsertPoint(*continuationBlock,
                         (*continuationBlock)->getFirstInsertionPt());
  return success();
}

namespace {
/// Owning equivalents of OpenMPIRBuilder::(Atomic)ReductionGen that are used to
/// store lambdas with capture.
using OwningReductionGen =
    std::function<llvm::OpenMPIRBuilder::InsertPointOrErrorTy(
        llvm::OpenMPIRBuilder::InsertPointTy, llvm::Value *, llvm::Value *,
        llvm::Value *&)>;
using OwningAtomicReductionGen =
    std::function<llvm::OpenMPIRBuilder::InsertPointOrErrorTy(
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
                llvm::Value *&result) mutable
      -> llvm::OpenMPIRBuilder::InsertPointOrErrorTy {
    moduleTranslation.mapValue(decl.getReductionLhsArg(), lhs);
    moduleTranslation.mapValue(decl.getReductionRhsArg(), rhs);
    builder.restoreIP(insertPoint);
    SmallVector<llvm::Value *> phis;
    if (failed(inlineConvertOmpRegions(decl.getReductionRegion(),
                                       "omp.reduction.nonatomic.body", builder,
                                       moduleTranslation, &phis)))
      return llvm::createStringError(
          "failed to inline `combiner` region of `omp.declare_reduction`");
    result = llvm::getSingleElement(phis);
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
                llvm::Value *lhs, llvm::Value *rhs) mutable
      -> llvm::OpenMPIRBuilder::InsertPointOrErrorTy {
    moduleTranslation.mapValue(decl.getAtomicReductionLhsArg(), lhs);
    moduleTranslation.mapValue(decl.getAtomicReductionRhsArg(), rhs);
    builder.restoreIP(insertPoint);
    SmallVector<llvm::Value *> phis;
    if (failed(inlineConvertOmpRegions(decl.getAtomicReductionRegion(),
                                       "omp.reduction.atomic.body", builder,
                                       moduleTranslation, &phis)))
      return llvm::createStringError(
          "failed to inline `atomic` region of `omp.declare_reduction`");
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

  if (failed(checkImplementationStatus(opInst)))
    return failure();

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

  if (failed(checkImplementationStatus(opInst)))
    return failure();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // OrderedOp has only one region associated with it.
    auto &region = cast<omp::OrderedRegionOp>(opInst).getRegion();
    builder.restoreIP(codeGenIP);
    return convertOmpOpRegions(region, "omp.ordered.region", builder,
                               moduleTranslation)
        .takeError();
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) { return llvm::Error::success(); };

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createOrderedThreadsSimd(
          ompLoc, bodyGenCB, finiCB, !orderedRegionOp.getParLevelSimd());

  if (failed(handleError(afterIP, opInst)))
    return failure();

  builder.restoreIP(*afterIP);
  return success();
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
                   const llvm::OpenMPIRBuilder::InsertPointTy &allocaIP,
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
        return loop.emitError(
            "failed to inline `alloc` region of `omp.declare_reduction`");

      assert(phis.size() == 1 && "expected one allocation to be yielded");
      builder.SetInsertPoint(allocaIP.getBlock()->getTerminator());

      // Allocate reduction variable (which is a pointer to the real reduction
      // variable allocated in the inlined region)
      llvm::Value *var = builder.CreateAlloca(
          moduleTranslation.convertType(reductionDecls[i].getType()));

      llvm::Type *ptrTy = builder.getPtrTy();
      llvm::Value *castVar =
          builder.CreatePointerBitCastOrAddrSpaceCast(var, ptrTy);
      llvm::Value *castPhi =
          builder.CreatePointerBitCastOrAddrSpaceCast(phis[0], ptrTy);

      deferredStores.emplace_back(castPhi, castVar);

      privateReductionVariables[i] = castVar;
      moduleTranslation.mapValue(reductionArgs[i], castPhi);
      reductionVariableMap.try_emplace(loop.getReductionVars()[i], castPhi);
    } else {
      assert(allocRegion.empty() &&
             "allocaction is implicit for by-val reduction");
      llvm::Value *var = builder.CreateAlloca(
          moduleTranslation.convertType(reductionDecls[i].getType()));

      llvm::Type *ptrTy = builder.getPtrTy();
      llvm::Value *castVar =
          builder.CreatePointerBitCastOrAddrSpaceCast(var, ptrTy);

      moduleTranslation.mapValue(reductionArgs[i], castVar);
      privateReductionVariables[i] = castVar;
      reductionVariableMap.try_emplace(loop.getReductionVars()[i], castVar);
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

static void
setInsertPointForPossiblyEmptyBlock(llvm::IRBuilderBase &builder,
                                    llvm::BasicBlock *block = nullptr) {
  if (block == nullptr)
    block = builder.GetInsertBlock();

  if (block->empty() || block->getTerminator() == nullptr)
    builder.SetInsertPoint(block);
  else
    builder.SetInsertPoint(block->getTerminator());
}

/// Inline reductions' `init` regions. This functions assumes that the
/// `builder`'s insertion point is where the user wants the `init` regions to be
/// inlined; i.e. it does not try to find a proper insertion location for the
/// `init` regions. It also leaves the `builder's insertions point in a state
/// where the user can continue the code-gen directly afterwards.
template <typename OP>
static LogicalResult
initReductionVars(OP op, ArrayRef<BlockArgument> reductionArgs,
                  llvm::IRBuilderBase &builder,
                  LLVM::ModuleTranslation &moduleTranslation,
                  llvm::BasicBlock *latestAllocaBlock,
                  SmallVectorImpl<omp::DeclareReductionOp> &reductionDecls,
                  SmallVectorImpl<llvm::Value *> &privateReductionVariables,
                  DenseMap<Value, llvm::Value *> &reductionVariableMap,
                  llvm::ArrayRef<bool> isByRef,
                  SmallVectorImpl<DeferredStore> &deferredStores) {
  if (op.getNumReductionVars() == 0)
    return success();

  llvm::BasicBlock *initBlock = splitBB(builder, true, "omp.reduction.init");
  auto allocaIP = llvm::IRBuilderBase::InsertPoint(
      latestAllocaBlock, latestAllocaBlock->getTerminator()->getIterator());
  builder.restoreIP(allocaIP);
  SmallVector<llvm::Value *> byRefVars(op.getNumReductionVars());

  for (unsigned i = 0; i < op.getNumReductionVars(); ++i) {
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

  setInsertPointForPossiblyEmptyBlock(builder, initBlock);

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

    setInsertPointForPossiblyEmptyBlock(builder);

    if (isByRef[i]) {
      if (!reductionDecls[i].getAllocRegion().empty())
        // done in allocReductionVars
        continue;

      // TODO: this path can be removed once all users of by-ref are updated to
      // use an alloc region

      // Store the result of the inlined region to the allocated reduction var
      // ptr
      builder.CreateStore(phis[0], byRefVars[i]);

      privateReductionVariables[i] = byRefVars[i];
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
    ArrayRef<llvm::Value *> privateReductionVariables, ArrayRef<bool> isByRef,
    bool isNowait = false, bool isTeamsReduction = false) {
  // Process the reductions if required.
  if (op.getNumReductionVars() == 0)
    return success();

  SmallVector<OwningReductionGen> owningReductionGens;
  SmallVector<OwningAtomicReductionGen> owningAtomicReductionGens;
  SmallVector<llvm::OpenMPIRBuilder::ReductionInfo> reductionInfos;

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // Create the reduction generators. We need to own them here because
  // ReductionInfo only accepts references to the generators.
  collectReductionInfo(op, builder, moduleTranslation, reductionDecls,
                       owningReductionGens, owningAtomicReductionGens,
                       privateReductionVariables, reductionInfos);

  // The call to createReductions below expects the block to have a
  // terminator. Create an unreachable instruction to serve as terminator
  // and remove it later.
  llvm::UnreachableInst *tempTerminator = builder.CreateUnreachable();
  builder.SetInsertPoint(tempTerminator);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy contInsertPoint =
      ompBuilder->createReductions(builder.saveIP(), allocaIP, reductionInfos,
                                   isByRef, isNowait, isTeamsReduction);

  if (failed(handleError(contInsertPoint, *op)))
    return failure();

  if (!contInsertPoint->getBlock())
    return op->emitOpError() << "failed to convert reductions";

  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      ompBuilder->createBarrier(*contInsertPoint, llvm::omp::OMPD_for);

  if (failed(handleError(afterIP, *op)))
    return failure();

  tempTerminator->eraseFromParent();
  builder.restoreIP(*afterIP);

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

  return initReductionVars(op, reductionArgs, builder, moduleTranslation,
                           allocaIP.getBlock(), reductionDecls,
                           privateReductionVariables, reductionVariableMap,
                           isByRef, deferredStores);
}

/// Return the llvm::Value * corresponding to the `privateVar` that
/// is being privatized. It isn't always as simple as looking up
/// moduleTranslation with privateVar. For instance, in case of
/// an allocatable, the descriptor for the allocatable is privatized.
/// This descriptor is mapped using an MapInfoOp. So, this function
/// will return a pointer to the llvm::Value corresponding to the
/// block argument for the mapped descriptor.
static llvm::Value *
findAssociatedValue(Value privateVar, llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    llvm::DenseMap<Value, Value> *mappedPrivateVars = nullptr) {
  if (mappedPrivateVars == nullptr || !mappedPrivateVars->contains(privateVar))
    return moduleTranslation.lookupValue(privateVar);

  Value blockArg = (*mappedPrivateVars)[privateVar];
  Type privVarType = privateVar.getType();
  Type blockArgType = blockArg.getType();
  assert(isa<LLVM::LLVMPointerType>(blockArgType) &&
         "A block argument corresponding to a mapped var should have "
         "!llvm.ptr type");

  if (privVarType == blockArgType)
    return moduleTranslation.lookupValue(blockArg);

  // This typically happens when the privatized type is lowered from
  // boxchar<KIND> and gets lowered to !llvm.struct<(ptr, i64)>. That is the
  // struct/pair is passed by value. But, mapped values are passed only as
  // pointers, so before we privatize, we must load the pointer.
  if (!isa<LLVM::LLVMPointerType>(privVarType))
    return builder.CreateLoad(moduleTranslation.convertType(privVarType),
                              moduleTranslation.lookupValue(blockArg));

  return moduleTranslation.lookupValue(privateVar);
}

/// Initialize a single (first)private variable. You probably want to use
/// allocateAndInitPrivateVars instead of this.
/// This returns the private variable which has been initialized. This
/// variable should be mapped before constructing the body of the Op.
static llvm::Expected<llvm::Value *> initPrivateVar(
    llvm::IRBuilderBase &builder, LLVM::ModuleTranslation &moduleTranslation,
    omp::PrivateClauseOp &privDecl, Value mlirPrivVar, BlockArgument &blockArg,
    llvm::Value *llvmPrivateVar, llvm::BasicBlock *privInitBlock,
    llvm::DenseMap<Value, Value> *mappedPrivateVars = nullptr) {
  Region &initRegion = privDecl.getInitRegion();
  if (initRegion.empty())
    return llvmPrivateVar;

  // map initialization region block arguments
  llvm::Value *nonPrivateVar = findAssociatedValue(
      mlirPrivVar, builder, moduleTranslation, mappedPrivateVars);
  assert(nonPrivateVar);
  moduleTranslation.mapValue(privDecl.getInitMoldArg(), nonPrivateVar);
  moduleTranslation.mapValue(privDecl.getInitPrivateArg(), llvmPrivateVar);

  // in-place convert the private initialization region
  SmallVector<llvm::Value *, 1> phis;
  if (failed(inlineConvertOmpRegions(initRegion, "omp.private.init", builder,
                                     moduleTranslation, &phis)))
    return llvm::createStringError(
        "failed to inline `init` region of `omp.private`");

  assert(phis.size() == 1 && "expected one allocation to be yielded");

  // clear init region block argument mapping in case it needs to be
  // re-created with a different source for another use of the same
  // reduction decl
  moduleTranslation.forgetMapping(initRegion);

  // Prefer the value yielded from the init region to the allocated private
  // variable in case the region is operating on arguments by-value (e.g.
  // Fortran character boxes).
  return phis[0];
}

static llvm::Error
initPrivateVars(llvm::IRBuilderBase &builder,
                LLVM::ModuleTranslation &moduleTranslation,
                PrivateVarsInfo &privateVarsInfo,
                llvm::DenseMap<Value, Value> *mappedPrivateVars = nullptr) {
  if (privateVarsInfo.blockArgs.empty())
    return llvm::Error::success();

  llvm::BasicBlock *privInitBlock = splitBB(builder, true, "omp.private.init");
  setInsertPointForPossiblyEmptyBlock(builder, privInitBlock);

  for (auto [idx, zip] : llvm::enumerate(llvm::zip_equal(
           privateVarsInfo.privatizers, privateVarsInfo.mlirVars,
           privateVarsInfo.blockArgs, privateVarsInfo.llvmVars))) {
    auto [privDecl, mlirPrivVar, blockArg, llvmPrivateVar] = zip;
    llvm::Expected<llvm::Value *> privVarOrErr = initPrivateVar(
        builder, moduleTranslation, privDecl, mlirPrivVar, blockArg,
        llvmPrivateVar, privInitBlock, mappedPrivateVars);

    if (!privVarOrErr)
      return privVarOrErr.takeError();

    llvmPrivateVar = privVarOrErr.get();
    moduleTranslation.mapValue(blockArg, llvmPrivateVar);

    setInsertPointForPossiblyEmptyBlock(builder);
  }

  return llvm::Error::success();
}

/// Allocate and initialize delayed private variables. Returns the basic block
/// which comes after all of these allocations. llvm::Value * for each of these
/// private variables are populated in llvmPrivateVars.
static llvm::Expected<llvm::BasicBlock *>
allocatePrivateVars(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    PrivateVarsInfo &privateVarsInfo,
                    const llvm::OpenMPIRBuilder::InsertPointTy &allocaIP,
                    llvm::DenseMap<Value, Value> *mappedPrivateVars = nullptr) {
  // Allocate private vars
  llvm::Instruction *allocaTerminator = allocaIP.getBlock()->getTerminator();
  splitBB(llvm::OpenMPIRBuilder::InsertPointTy(allocaIP.getBlock(),
                                               allocaTerminator->getIterator()),
          true, allocaTerminator->getStableDebugLoc(),
          "omp.region.after_alloca");

  llvm::IRBuilderBase::InsertPointGuard guard(builder);
  // Update the allocaTerminator since the alloca block was split above.
  allocaTerminator = allocaIP.getBlock()->getTerminator();
  builder.SetInsertPoint(allocaTerminator);
  // The new terminator is an uncondition branch created by the splitBB above.
  assert(allocaTerminator->getNumSuccessors() == 1 &&
         "This is an unconditional branch created by splitBB");

  llvm::DataLayout dataLayout = builder.GetInsertBlock()->getDataLayout();
  llvm::BasicBlock *afterAllocas = allocaTerminator->getSuccessor(0);

  unsigned int allocaAS =
      moduleTranslation.getLLVMModule()->getDataLayout().getAllocaAddrSpace();
  unsigned int defaultAS = moduleTranslation.getLLVMModule()
                               ->getDataLayout()
                               .getProgramAddressSpace();

  for (auto [privDecl, mlirPrivVar, blockArg] :
       llvm::zip_equal(privateVarsInfo.privatizers, privateVarsInfo.mlirVars,
                       privateVarsInfo.blockArgs)) {
    llvm::Type *llvmAllocType =
        moduleTranslation.convertType(privDecl.getType());
    builder.SetInsertPoint(allocaIP.getBlock()->getTerminator());
    llvm::Value *llvmPrivateVar = builder.CreateAlloca(
        llvmAllocType, /*ArraySize=*/nullptr, "omp.private.alloc");
    if (allocaAS != defaultAS)
      llvmPrivateVar = builder.CreateAddrSpaceCast(llvmPrivateVar,
                                                   builder.getPtrTy(defaultAS));

    privateVarsInfo.llvmVars.push_back(llvmPrivateVar);
  }

  return afterAllocas;
}

static LogicalResult copyFirstPrivateVars(
    mlir::Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation,
    SmallVectorImpl<mlir::Value> &mlirPrivateVars,
    ArrayRef<llvm::Value *> llvmPrivateVars,
    SmallVectorImpl<omp::PrivateClauseOp> &privateDecls, bool insertBarrier,
    llvm::DenseMap<Value, Value> *mappedPrivateVars = nullptr) {
  // Apply copy region for firstprivate.
  bool needsFirstprivate =
      llvm::any_of(privateDecls, [](omp::PrivateClauseOp &privOp) {
        return privOp.getDataSharingType() ==
               omp::DataSharingClauseType::FirstPrivate;
      });

  if (!needsFirstprivate)
    return success();

  llvm::BasicBlock *copyBlock =
      splitBB(builder, /*CreateBranch=*/true, "omp.private.copy");
  setInsertPointForPossiblyEmptyBlock(builder, copyBlock);

  for (auto [decl, mlirVar, llvmVar] :
       llvm::zip_equal(privateDecls, mlirPrivateVars, llvmPrivateVars)) {
    if (decl.getDataSharingType() != omp::DataSharingClauseType::FirstPrivate)
      continue;

    // copyRegion implements `lhs = rhs`
    Region &copyRegion = decl.getCopyRegion();

    // map copyRegion rhs arg
    llvm::Value *nonPrivateVar = findAssociatedValue(
        mlirVar, builder, moduleTranslation, mappedPrivateVars);
    assert(nonPrivateVar);
    moduleTranslation.mapValue(decl.getCopyMoldArg(), nonPrivateVar);

    // map copyRegion lhs arg
    moduleTranslation.mapValue(decl.getCopyPrivateArg(), llvmVar);

    // in-place convert copy region
    if (failed(inlineConvertOmpRegions(copyRegion, "omp.private.copy", builder,
                                       moduleTranslation)))
      return decl.emitError("failed to inline `copy` region of `omp.private`");

    setInsertPointForPossiblyEmptyBlock(builder);

    // ignore unused value yielded from copy region

    // clear copy region block argument mapping in case it needs to be
    // re-created with different sources for reuse of the same reduction
    // decl
    moduleTranslation.forgetMapping(copyRegion);
  }

  if (insertBarrier) {
    llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
    llvm::OpenMPIRBuilder::InsertPointOrErrorTy res =
        ompBuilder->createBarrier(builder.saveIP(), llvm::omp::OMPD_barrier);
    if (failed(handleError(res, *op)))
      return failure();
  }

  return success();
}

static LogicalResult
cleanupPrivateVars(llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation, Location loc,
                   SmallVectorImpl<llvm::Value *> &llvmPrivateVars,
                   SmallVectorImpl<omp::PrivateClauseOp> &privateDecls) {
  // private variable deallocation
  SmallVector<Region *> privateCleanupRegions;
  llvm::transform(privateDecls, std::back_inserter(privateCleanupRegions),
                  [](omp::PrivateClauseOp privatizer) {
                    return &privatizer.getDeallocRegion();
                  });

  if (failed(inlineOmpRegionCleanup(
          privateCleanupRegions, llvmPrivateVars, moduleTranslation, builder,
          "omp.private.dealloc", /*shouldLoadCleanupRegionArg=*/false)))
    return mlir::emitError(loc, "failed to inline `dealloc` region of an "
                                "`omp.private` op in");

  return success();
}

/// Returns true if the construct contains omp.cancel or omp.cancellation_point
static bool constructIsCancellable(Operation *op) {
  // omp.cancel and omp.cancellation_point must be "closely nested" so they will
  // be visible and not inside of function calls. This is enforced by the
  // verifier.
  return op
      ->walk([](Operation *child) {
        if (mlir::isa<omp::CancelOp, omp::CancellationPointOp>(child))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static LogicalResult
convertOmpSections(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  using StorableBodyGenCallbackTy =
      llvm::OpenMPIRBuilder::StorableBodyGenCallbackTy;

  auto sectionsOp = cast<omp::SectionsOp>(opInst);

  if (failed(checkImplementationStatus(opInst)))
    return failure();

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
      cast<omp::BlockArgOpenMPOpInterface>(opInst).getReductionBlockArgs();

  if (failed(allocAndInitializeReductionVars(
          sectionsOp, reductionArgs, builder, moduleTranslation, allocaIP,
          reductionDecls, privateReductionVariables, reductionVariableMap,
          isByRef)))
    return failure();

  SmallVector<StorableBodyGenCallbackTy> sectionCBs;

  for (Operation &op : *sectionsOp.getRegion().begin()) {
    auto sectionOp = dyn_cast<omp::SectionOp>(op);
    if (!sectionOp) // omp.terminator
      continue;

    Region &region = sectionOp.getRegion();
    auto sectionCB = [&sectionsOp, &region, &builder, &moduleTranslation](
                         InsertPointTy allocaIP, InsertPointTy codeGenIP) {
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

      return convertOmpOpRegions(region, "omp.section.region", builder,
                                 moduleTranslation)
          .takeError();
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
                    llvm::Value &vPtr, llvm::Value *&replacementValue)
      -> llvm::OpenMPIRBuilder::InsertPointOrErrorTy {
    replacementValue = &vPtr;
    return codeGenIP;
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) { return llvm::Error::success(); };

  allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  bool isCancellable = constructIsCancellable(sectionsOp);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createSections(
          ompLoc, allocaIP, sectionCBs, privCB, finiCB, isCancellable,
          sectionsOp.getNowait());

  if (failed(handleError(afterIP, opInst)))
    return failure();

  builder.restoreIP(*afterIP);

  // Process the reductions if required.
  return createReductionsAndCleanup(
      sectionsOp, builder, moduleTranslation, allocaIP, reductionDecls,
      privateReductionVariables, isByRef, sectionsOp.getNowait());
}

/// Converts an OpenMP single construct into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpSingle(omp::SingleOp &singleOp, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  if (failed(checkImplementationStatus(*singleOp)))
    return failure();

  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    builder.restoreIP(codegenIP);
    return convertOmpOpRegions(singleOp.getRegion(), "omp.single.region",
                               builder, moduleTranslation)
        .takeError();
  };
  auto finiCB = [&](InsertPointTy codeGenIP) { return llvm::Error::success(); };

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

  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createSingle(
          ompLoc, bodyCB, finiCB, singleOp.getNowait(), llvmCPVars,
          llvmCPFuncs);

  if (failed(handleError(afterIP, *singleOp)))
    return failure();

  builder.restoreIP(*afterIP);
  return success();
}

static bool teamsReductionContainedInDistribute(omp::TeamsOp teamsOp) {
  auto iface =
      llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(teamsOp.getOperation());
  // Check that all uses of the reduction block arg has the same distribute op
  // parent.
  llvm::SmallVector<mlir::Operation *> debugUses;
  Operation *distOp = nullptr;
  for (auto ra : iface.getReductionBlockArgs())
    for (auto &use : ra.getUses()) {
      auto *useOp = use.getOwner();
      // Ignore debug uses.
      if (mlir::isa<LLVM::DbgDeclareOp, LLVM::DbgValueOp>(useOp)) {
        debugUses.push_back(useOp);
        continue;
      }

      auto currentDistOp = useOp->getParentOfType<omp::DistributeOp>();
      // Use is not inside a distribute op - return false
      if (!currentDistOp)
        return false;
      // Multiple distribute operations - return false
      Operation *currentOp = currentDistOp.getOperation();
      if (distOp && (distOp != currentOp))
        return false;

      distOp = currentOp;
    }

  // If we are going to use distribute reduction then remove any debug uses of
  // the reduction parameters in teamsOp. Otherwise they will be left without
  // any mapped value in moduleTranslation and will eventually error out.
  for (auto use : debugUses)
    use->erase();
  return true;
}

// Convert an OpenMP Teams construct to LLVM IR using OpenMPIRBuilder
static LogicalResult
convertOmpTeams(omp::TeamsOp op, llvm::IRBuilderBase &builder,
                LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  if (failed(checkImplementationStatus(*op)))
    return failure();

  DenseMap<Value, llvm::Value *> reductionVariableMap;
  unsigned numReductionVars = op.getNumReductionVars();
  SmallVector<omp::DeclareReductionOp> reductionDecls;
  SmallVector<llvm::Value *> privateReductionVariables(numReductionVars);
  llvm::ArrayRef<bool> isByRef;
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  // Only do teams reduction if there is no distribute op that captures the
  // reduction instead.
  bool doTeamsReduction = !teamsReductionContainedInDistribute(op);
  if (doTeamsReduction) {
    isByRef = getIsByRef(op.getReductionByref());

    assert(isByRef.size() == op.getNumReductionVars());

    MutableArrayRef<BlockArgument> reductionArgs =
        llvm::cast<omp::BlockArgOpenMPOpInterface>(*op).getReductionBlockArgs();

    collectReductionDecls(op, reductionDecls);

    if (failed(allocAndInitializeReductionVars(
            op, reductionArgs, builder, moduleTranslation, allocaIP,
            reductionDecls, privateReductionVariables, reductionVariableMap,
            isByRef)))
      return failure();
  }

  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
        moduleTranslation, allocaIP);
    builder.restoreIP(codegenIP);
    return convertOmpOpRegions(op.getRegion(), "omp.teams.region", builder,
                               moduleTranslation)
        .takeError();
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
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createTeams(
          ompLoc, bodyCB, numTeamsLower, numTeamsUpper, threadLimit, ifExpr);

  if (failed(handleError(afterIP, *op)))
    return failure();

  builder.restoreIP(*afterIP);
  if (doTeamsReduction) {
    // Process the reductions if required.
    return createReductionsAndCleanup(
        op, builder, moduleTranslation, allocaIP, reductionDecls,
        privateReductionVariables, isByRef,
        /*isNoWait*/ false, /*isTeamsReduction*/ true);
  }
  return success();
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
    case mlir::omp::ClauseTaskDepend::taskdependmutexinoutset:
      type = llvm::omp::RTLDependenceKindTy::DepMutexInOutSet;
      break;
    case mlir::omp::ClauseTaskDepend::taskdependinoutset:
      type = llvm::omp::RTLDependenceKindTy::DepInOutSet;
      break;
    };
    llvm::Value *depVal = moduleTranslation.lookupValue(std::get<0>(dep));
    llvm::OpenMPIRBuilder::DependData dd(type, depVal->getType(), depVal);
    dds.emplace_back(dd);
  }
}

/// Shared implementation of a callback which adds a termiator for the new block
/// created for the branch taken when an openmp construct is cancelled. The
/// terminator is saved in \p cancelTerminators. This callback is invoked only
/// if there is cancellation inside of the taskgroup body.
/// The terminator will need to be fixed to branch to the correct block to
/// cleanup the construct.
static void
pushCancelFinalizationCB(SmallVectorImpl<llvm::BranchInst *> &cancelTerminators,
                         llvm::IRBuilderBase &llvmBuilder,
                         llvm::OpenMPIRBuilder &ompBuilder, mlir::Operation *op,
                         llvm::omp::Directive cancelDirective) {
  auto finiCB = [&](llvm::OpenMPIRBuilder::InsertPointTy ip) -> llvm::Error {
    llvm::IRBuilderBase::InsertPointGuard guard(llvmBuilder);

    // ip is currently in the block branched to if cancellation occured.
    // We need to create a branch to terminate that block.
    llvmBuilder.restoreIP(ip);

    // We must still clean up the construct after cancelling it, so we need to
    // branch to the block that finalizes the taskgroup.
    // That block has not been created yet so use this block as a dummy for now
    // and fix this after creating the operation.
    cancelTerminators.push_back(llvmBuilder.CreateBr(ip.getBlock()));
    return llvm::Error::success();
  };
  // We have to add the cleanup to the OpenMPIRBuilder before the body gets
  // created in case the body contains omp.cancel (which will then expect to be
  // able to find this cleanup callback).
  ompBuilder.pushFinalizationCB(
      {finiCB, cancelDirective, constructIsCancellable(op)});
}

/// If we cancelled the construct, we should branch to the finalization block of
/// that construct. OMPIRBuilder structures the CFG such that the cleanup block
/// is immediately before the continuation block. Now this finalization has
/// been created we can fix the branch.
static void
popCancelFinalizationCB(const ArrayRef<llvm::BranchInst *> cancelTerminators,
                        llvm::OpenMPIRBuilder &ompBuilder,
                        const llvm::OpenMPIRBuilder::InsertPointTy &afterIP) {
  ompBuilder.popFinalizationCB();
  llvm::BasicBlock *constructFini = afterIP.getBlock()->getSinglePredecessor();
  for (llvm::BranchInst *cancelBranch : cancelTerminators) {
    assert(cancelBranch->getNumSuccessors() == 1 &&
           "cancel branch should have one target");
    cancelBranch->setSuccessor(0, constructFini);
  }
}

namespace {
/// TaskContextStructManager takes care of creating and freeing a structure
/// containing information needed by the task body to execute.
class TaskContextStructManager {
public:
  TaskContextStructManager(llvm::IRBuilderBase &builder,
                           LLVM::ModuleTranslation &moduleTranslation,
                           MutableArrayRef<omp::PrivateClauseOp> privateDecls)
      : builder{builder}, moduleTranslation{moduleTranslation},
        privateDecls{privateDecls} {}

  /// Creates a heap allocated struct containing space for each private
  /// variable. Invariant: privateVarTypes, privateDecls, and the elements of
  /// the structure should all have the same order (although privateDecls which
  /// do not read from the mold argument are skipped).
  void generateTaskContextStruct();

  /// Create GEPs to access each member of the structure representing a private
  /// variable, adding them to llvmPrivateVars. Null values are added where
  /// private decls were skipped so that the ordering continues to match the
  /// private decls.
  void createGEPsToPrivateVars();

  /// De-allocate the task context structure.
  void freeStructPtr();

  MutableArrayRef<llvm::Value *> getLLVMPrivateVarGEPs() {
    return llvmPrivateVarGEPs;
  }

  llvm::Value *getStructPtr() { return structPtr; }

private:
  llvm::IRBuilderBase &builder;
  LLVM::ModuleTranslation &moduleTranslation;
  MutableArrayRef<omp::PrivateClauseOp> privateDecls;

  /// The type of each member of the structure, in order.
  SmallVector<llvm::Type *> privateVarTypes;

  /// LLVM values for each private variable, or null if that private variable is
  /// not included in the task context structure
  SmallVector<llvm::Value *> llvmPrivateVarGEPs;

  /// A pointer to the structure containing context for this task.
  llvm::Value *structPtr = nullptr;
  /// The type of the structure
  llvm::Type *structTy = nullptr;
};
} // namespace

void TaskContextStructManager::generateTaskContextStruct() {
  if (privateDecls.empty())
    return;
  privateVarTypes.reserve(privateDecls.size());

  for (omp::PrivateClauseOp &privOp : privateDecls) {
    // Skip private variables which can safely be allocated and initialised
    // inside of the task
    if (!privOp.readsFromMold())
      continue;
    Type mlirType = privOp.getType();
    privateVarTypes.push_back(moduleTranslation.convertType(mlirType));
  }

  structTy = llvm::StructType::get(moduleTranslation.getLLVMContext(),
                                   privateVarTypes);

  llvm::DataLayout dataLayout =
      builder.GetInsertBlock()->getModule()->getDataLayout();
  llvm::Type *intPtrTy = builder.getIntPtrTy(dataLayout);
  llvm::Constant *allocSize = llvm::ConstantExpr::getSizeOf(structTy);

  // Heap allocate the structure
  structPtr = builder.CreateMalloc(intPtrTy, structTy, allocSize,
                                   /*ArraySize=*/nullptr, /*MallocF=*/nullptr,
                                   "omp.task.context_ptr");
}

void TaskContextStructManager::createGEPsToPrivateVars() {
  if (!structPtr) {
    assert(privateVarTypes.empty());
    return;
  }

  // Create GEPs for each struct member
  llvmPrivateVarGEPs.clear();
  llvmPrivateVarGEPs.reserve(privateDecls.size());
  llvm::Value *zero = builder.getInt32(0);
  unsigned i = 0;
  for (auto privDecl : privateDecls) {
    if (!privDecl.readsFromMold()) {
      // Handle this inside of the task so we don't pass unnessecary vars in
      llvmPrivateVarGEPs.push_back(nullptr);
      continue;
    }
    llvm::Value *iVal = builder.getInt32(i);
    llvm::Value *gep = builder.CreateGEP(structTy, structPtr, {zero, iVal});
    llvmPrivateVarGEPs.push_back(gep);
    i += 1;
  }
}

void TaskContextStructManager::freeStructPtr() {
  if (!structPtr)
    return;

  llvm::IRBuilderBase::InsertPointGuard guard{builder};
  // Ensure we don't put the call to free() after the terminator
  builder.SetInsertPoint(builder.GetInsertBlock()->getTerminator());
  builder.CreateFree(structPtr);
}

/// Converts an OpenMP task construct into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpTaskOp(omp::TaskOp taskOp, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  if (failed(checkImplementationStatus(*taskOp)))
    return failure();

  PrivateVarsInfo privateVarsInfo(taskOp);
  TaskContextStructManager taskStructMgr{builder, moduleTranslation,
                                         privateVarsInfo.privatizers};

  // Allocate and copy private variables before creating the task. This avoids
  // accessing invalid memory if (after this scope ends) the private variables
  // are initialized from host variables or if the variables are copied into
  // from host variables (firstprivate). The insertion point is just before
  // where the code for creating and scheduling the task will go. That puts this
  // code outside of the outlined task region, which is what we want because
  // this way the initialization and copy regions are executed immediately while
  // the host variable data are still live.

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  // Not using splitBB() because that requires the current block to have a
  // terminator.
  assert(builder.GetInsertPoint() == builder.GetInsertBlock()->end());
  llvm::BasicBlock *taskStartBlock = llvm::BasicBlock::Create(
      builder.getContext(), "omp.task.start",
      /*Parent=*/builder.GetInsertBlock()->getParent());
  llvm::Instruction *branchToTaskStartBlock = builder.CreateBr(taskStartBlock);
  builder.SetInsertPoint(branchToTaskStartBlock);

  // Now do this again to make the initialization and copy blocks
  llvm::BasicBlock *copyBlock =
      splitBB(builder, /*CreateBranch=*/true, "omp.private.copy");
  llvm::BasicBlock *initBlock =
      splitBB(builder, /*CreateBranch=*/true, "omp.private.init");

  // Now the control flow graph should look like
  // starter_block:
  //   <---- where we started when convertOmpTaskOp was called
  //   br %omp.private.init
  // omp.private.init:
  //   br %omp.private.copy
  // omp.private.copy:
  //   br %omp.task.start
  // omp.task.start:
  //   <---- where we want the insertion point to be when we call createTask()

  // Save the alloca insertion point on ModuleTranslation stack for use in
  // nested regions.
  LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
      moduleTranslation, allocaIP);

  // Allocate and initialize private variables
  builder.SetInsertPoint(initBlock->getTerminator());

  // Create task variable structure
  taskStructMgr.generateTaskContextStruct();
  // GEPs so that we can initialize the variables. Don't use these GEPs inside
  // of the body otherwise it will be the GEP not the struct which is fowarded
  // to the outlined function. GEPs forwarded in this way are passed in a
  // stack-allocated (by OpenMPIRBuilder) structure which is not safe for tasks
  // which may not be executed until after the current stack frame goes out of
  // scope.
  taskStructMgr.createGEPsToPrivateVars();

  for (auto [privDecl, mlirPrivVar, blockArg, llvmPrivateVarAlloc] :
       llvm::zip_equal(privateVarsInfo.privatizers, privateVarsInfo.mlirVars,
                       privateVarsInfo.blockArgs,
                       taskStructMgr.getLLVMPrivateVarGEPs())) {
    // To be handled inside the task.
    if (!privDecl.readsFromMold())
      continue;
    assert(llvmPrivateVarAlloc &&
           "reads from mold so shouldn't have been skipped");

    llvm::Expected<llvm::Value *> privateVarOrErr =
        initPrivateVar(builder, moduleTranslation, privDecl, mlirPrivVar,
                       blockArg, llvmPrivateVarAlloc, initBlock);
    if (!privateVarOrErr)
      return handleError(privateVarOrErr, *taskOp.getOperation());

    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    builder.SetInsertPoint(builder.GetInsertBlock()->getTerminator());

    // TODO: this is a bit of a hack for Fortran character boxes.
    // Character boxes are passed by value into the init region and then the
    // initialized character box is yielded by value. Here we need to store the
    // yielded value into the private allocation, and load the private
    // allocation to match the type expected by region block arguments.
    if ((privateVarOrErr.get() != llvmPrivateVarAlloc) &&
        !mlir::isa<LLVM::LLVMPointerType>(blockArg.getType())) {
      builder.CreateStore(privateVarOrErr.get(), llvmPrivateVarAlloc);
      // Load it so we have the value pointed to by the GEP
      llvmPrivateVarAlloc = builder.CreateLoad(privateVarOrErr.get()->getType(),
                                               llvmPrivateVarAlloc);
    }
    assert(llvmPrivateVarAlloc->getType() ==
           moduleTranslation.convertType(blockArg.getType()));

    // Mapping blockArg -> llvmPrivateVarAlloc is done inside the body callback
    // so that OpenMPIRBuilder doesn't try to pass each GEP address through a
    // stack allocated structure.
  }

  // firstprivate copy region
  setInsertPointForPossiblyEmptyBlock(builder, copyBlock);
  if (failed(copyFirstPrivateVars(
          taskOp, builder, moduleTranslation, privateVarsInfo.mlirVars,
          taskStructMgr.getLLVMPrivateVarGEPs(), privateVarsInfo.privatizers,
          taskOp.getPrivateNeedsBarrier())))
    return llvm::failure();

  // Set up for call to createTask()
  builder.SetInsertPoint(taskStartBlock);

  auto bodyCB = [&](InsertPointTy allocaIP,
                    InsertPointTy codegenIP) -> llvm::Error {
    // Save the alloca insertion point on ModuleTranslation stack for use in
    // nested regions.
    LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
        moduleTranslation, allocaIP);

    // translate the body of the task:
    builder.restoreIP(codegenIP);

    llvm::BasicBlock *privInitBlock = nullptr;
    privateVarsInfo.llvmVars.resize(privateVarsInfo.blockArgs.size());
    for (auto [i, zip] : llvm::enumerate(llvm::zip_equal(
             privateVarsInfo.blockArgs, privateVarsInfo.privatizers,
             privateVarsInfo.mlirVars))) {
      auto [blockArg, privDecl, mlirPrivVar] = zip;
      // This is handled before the task executes
      if (privDecl.readsFromMold())
        continue;

      llvm::IRBuilderBase::InsertPointGuard guard(builder);
      llvm::Type *llvmAllocType =
          moduleTranslation.convertType(privDecl.getType());
      builder.SetInsertPoint(allocaIP.getBlock()->getTerminator());
      llvm::Value *llvmPrivateVar = builder.CreateAlloca(
          llvmAllocType, /*ArraySize=*/nullptr, "omp.private.alloc");

      llvm::Expected<llvm::Value *> privateVarOrError =
          initPrivateVar(builder, moduleTranslation, privDecl, mlirPrivVar,
                         blockArg, llvmPrivateVar, privInitBlock);
      if (!privateVarOrError)
        return privateVarOrError.takeError();
      moduleTranslation.mapValue(blockArg, privateVarOrError.get());
      privateVarsInfo.llvmVars[i] = privateVarOrError.get();
    }

    taskStructMgr.createGEPsToPrivateVars();
    for (auto [i, llvmPrivVar] :
         llvm::enumerate(taskStructMgr.getLLVMPrivateVarGEPs())) {
      if (!llvmPrivVar) {
        assert(privateVarsInfo.llvmVars[i] &&
               "This is added in the loop above");
        continue;
      }
      privateVarsInfo.llvmVars[i] = llvmPrivVar;
    }

    // Find and map the addresses of each variable within the task context
    // structure
    for (auto [blockArg, llvmPrivateVar, privateDecl] :
         llvm::zip_equal(privateVarsInfo.blockArgs, privateVarsInfo.llvmVars,
                         privateVarsInfo.privatizers)) {
      // This was handled above.
      if (!privateDecl.readsFromMold())
        continue;
      // Fix broken pass-by-value case for Fortran character boxes
      if (!mlir::isa<LLVM::LLVMPointerType>(blockArg.getType())) {
        llvmPrivateVar = builder.CreateLoad(
            moduleTranslation.convertType(blockArg.getType()), llvmPrivateVar);
      }
      assert(llvmPrivateVar->getType() ==
             moduleTranslation.convertType(blockArg.getType()));
      moduleTranslation.mapValue(blockArg, llvmPrivateVar);
    }

    auto continuationBlockOrError = convertOmpOpRegions(
        taskOp.getRegion(), "omp.task.region", builder, moduleTranslation);
    if (failed(handleError(continuationBlockOrError, *taskOp)))
      return llvm::make_error<PreviouslyReportedError>();

    builder.SetInsertPoint(continuationBlockOrError.get()->getTerminator());

    if (failed(cleanupPrivateVars(builder, moduleTranslation, taskOp.getLoc(),
                                  privateVarsInfo.llvmVars,
                                  privateVarsInfo.privatizers)))
      return llvm::make_error<PreviouslyReportedError>();

    // Free heap allocated task context structure at the end of the task.
    taskStructMgr.freeStructPtr();

    return llvm::Error::success();
  };

  llvm::OpenMPIRBuilder &ompBuilder = *moduleTranslation.getOpenMPBuilder();
  SmallVector<llvm::BranchInst *> cancelTerminators;
  // The directive to match here is OMPD_taskgroup because it is the taskgroup
  // which is canceled. This is handled here because it is the task's cleanup
  // block which should be branched to.
  pushCancelFinalizationCB(cancelTerminators, builder, ompBuilder, taskOp,
                           llvm::omp::Directive::OMPD_taskgroup);

  SmallVector<llvm::OpenMPIRBuilder::DependData> dds;
  buildDependData(taskOp.getDependKinds(), taskOp.getDependVars(),
                  moduleTranslation, dds);

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createTask(
          ompLoc, allocaIP, bodyCB, !taskOp.getUntied(),
          moduleTranslation.lookupValue(taskOp.getFinal()),
          moduleTranslation.lookupValue(taskOp.getIfExpr()), dds,
          taskOp.getMergeable(),
          moduleTranslation.lookupValue(taskOp.getEventHandle()),
          moduleTranslation.lookupValue(taskOp.getPriority()));

  if (failed(handleError(afterIP, *taskOp)))
    return failure();

  // Set the correct branch target for task cancellation
  popCancelFinalizationCB(cancelTerminators, ompBuilder, afterIP.get());

  builder.restoreIP(*afterIP);
  return success();
}

/// Converts an OpenMP taskgroup construct into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpTaskgroupOp(omp::TaskgroupOp tgOp, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  if (failed(checkImplementationStatus(*tgOp)))
    return failure();

  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    builder.restoreIP(codegenIP);
    return convertOmpOpRegions(tgOp.getRegion(), "omp.taskgroup.region",
                               builder, moduleTranslation)
        .takeError();
  };

  InsertPointTy allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createTaskgroup(ompLoc, allocaIP,
                                                            bodyCB);

  if (failed(handleError(afterIP, *tgOp)))
    return failure();

  builder.restoreIP(*afterIP);
  return success();
}

static LogicalResult
convertOmpTaskwaitOp(omp::TaskwaitOp twOp, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {
  if (failed(checkImplementationStatus(*twOp)))
    return failure();

  moduleTranslation.getOpenMPBuilder()->createTaskwait(builder.saveIP());
  return success();
}

/// Converts an OpenMP workshare loop into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpWsloop(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  auto wsloopOp = cast<omp::WsloopOp>(opInst);
  if (failed(checkImplementationStatus(opInst)))
    return failure();

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

  PrivateVarsInfo privateVarsInfo(wsloopOp);

  SmallVector<omp::DeclareReductionOp> reductionDecls;
  collectReductionDecls(wsloopOp, reductionDecls);
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  SmallVector<llvm::Value *> privateReductionVariables(
      wsloopOp.getNumReductionVars());

  llvm::Expected<llvm::BasicBlock *> afterAllocas = allocatePrivateVars(
      builder, moduleTranslation, privateVarsInfo, allocaIP);
  if (handleError(afterAllocas, opInst).failed())
    return failure();

  DenseMap<Value, llvm::Value *> reductionVariableMap;

  MutableArrayRef<BlockArgument> reductionArgs =
      cast<omp::BlockArgOpenMPOpInterface>(opInst).getReductionBlockArgs();

  SmallVector<DeferredStore> deferredStores;

  if (failed(allocReductionVars(wsloopOp, reductionArgs, builder,
                                moduleTranslation, allocaIP, reductionDecls,
                                privateReductionVariables, reductionVariableMap,
                                deferredStores, isByRef)))
    return failure();

  if (handleError(initPrivateVars(builder, moduleTranslation, privateVarsInfo),
                  opInst)
          .failed())
    return failure();

  if (failed(copyFirstPrivateVars(
          wsloopOp, builder, moduleTranslation, privateVarsInfo.mlirVars,
          privateVarsInfo.llvmVars, privateVarsInfo.privatizers,
          wsloopOp.getPrivateNeedsBarrier())))
    return failure();

  assert(afterAllocas.get()->getSinglePredecessor());
  if (failed(initReductionVars(wsloopOp, reductionArgs, builder,
                               moduleTranslation,
                               afterAllocas.get()->getSinglePredecessor(),
                               reductionDecls, privateReductionVariables,
                               reductionVariableMap, isByRef, deferredStores)))
    return failure();

  // TODO: Handle doacross loops when the ordered clause has a parameter.
  bool isOrdered = wsloopOp.getOrdered().has_value();
  std::optional<omp::ScheduleModifier> scheduleMod = wsloopOp.getScheduleMod();
  bool isSimd = wsloopOp.getScheduleSimd();
  bool loopNeedsBarrier = !wsloopOp.getNowait();

  // The only legal way for the direct parent to be omp.distribute is that this
  // represents 'distribute parallel do'. Otherwise, this is a regular
  // worksharing loop.
  llvm::omp::WorksharingLoopType workshareLoopType =
      llvm::isa_and_present<omp::DistributeOp>(opInst.getParentOp())
          ? llvm::omp::WorksharingLoopType::DistributeForStaticLoop
          : llvm::omp::WorksharingLoopType::ForStaticLoop;

  SmallVector<llvm::BranchInst *> cancelTerminators;
  pushCancelFinalizationCB(cancelTerminators, builder, *ompBuilder, wsloopOp,
                           llvm::omp::Directive::OMPD_for);

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  // Initialize linear variables and linear step
  LinearClauseProcessor linearClauseProcessor;
  if (wsloopOp.getLinearVars().size()) {
    for (mlir::Value linearVar : wsloopOp.getLinearVars())
      linearClauseProcessor.createLinearVar(builder, moduleTranslation,
                                            linearVar);
    for (mlir::Value linearStep : wsloopOp.getLinearStepVars())
      linearClauseProcessor.initLinearStep(moduleTranslation, linearStep);
  }

  llvm::Expected<llvm::BasicBlock *> regionBlock = convertOmpOpRegions(
      wsloopOp.getRegion(), "omp.wsloop.region", builder, moduleTranslation);

  if (failed(handleError(regionBlock, opInst)))
    return failure();

  llvm::CanonicalLoopInfo *loopInfo = findCurrentLoopInfo(moduleTranslation);

  // Emit Initialization and Update IR for linear variables
  if (wsloopOp.getLinearVars().size()) {
    llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterBarrierIP =
        linearClauseProcessor.initLinearVar(builder, moduleTranslation,
                                            loopInfo->getPreheader());
    if (failed(handleError(afterBarrierIP, *loopOp)))
      return failure();
    builder.restoreIP(*afterBarrierIP);
    linearClauseProcessor.updateLinearVar(builder, loopInfo->getBody(),
                                          loopInfo->getIndVar());
    linearClauseProcessor.outlineLinearFinalizationBB(builder,
                                                      loopInfo->getExit());
  }

  builder.SetInsertPoint(*regionBlock, (*regionBlock)->begin());
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy wsloopIP =
      ompBuilder->applyWorkshareLoop(
          ompLoc.DL, loopInfo, allocaIP, loopNeedsBarrier,
          convertToScheduleKind(schedule), chunk, isSimd,
          scheduleMod == omp::ScheduleModifier::monotonic,
          scheduleMod == omp::ScheduleModifier::nonmonotonic, isOrdered,
          workshareLoopType);

  if (failed(handleError(wsloopIP, opInst)))
    return failure();

  // Emit finalization and in-place rewrites for linear vars.
  if (wsloopOp.getLinearVars().size()) {
    llvm::OpenMPIRBuilder::InsertPointTy oldIP = builder.saveIP();
    assert(loopInfo->getLastIter() &&
           "`lastiter` in CanonicalLoopInfo is nullptr");
    llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterBarrierIP =
        linearClauseProcessor.finalizeLinearVar(builder, moduleTranslation,
                                                loopInfo->getLastIter());
    if (failed(handleError(afterBarrierIP, *loopOp)))
      return failure();
    for (size_t index = 0; index < wsloopOp.getLinearVars().size(); index++)
      linearClauseProcessor.rewriteInPlace(builder, "omp.loop_nest.region",
                                           index);
    builder.restoreIP(oldIP);
  }

  // Set the correct branch target for task cancellation
  popCancelFinalizationCB(cancelTerminators, *ompBuilder, wsloopIP.get());

  // Process the reductions if required.
  if (failed(createReductionsAndCleanup(
          wsloopOp, builder, moduleTranslation, allocaIP, reductionDecls,
          privateReductionVariables, isByRef, wsloopOp.getNowait(),
          /*isTeamsReduction=*/false)))
    return failure();

  return cleanupPrivateVars(builder, moduleTranslation, wsloopOp.getLoc(),
                            privateVarsInfo.llvmVars,
                            privateVarsInfo.privatizers);
}

/// Converts the OpenMP parallel operation to LLVM IR.
static LogicalResult
convertOmpParallel(omp::ParallelOp opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  ArrayRef<bool> isByRef = getIsByRef(opInst.getReductionByref());
  assert(isByRef.size() == opInst.getNumReductionVars());
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  if (failed(checkImplementationStatus(*opInst)))
    return failure();

  PrivateVarsInfo privateVarsInfo(opInst);

  // Collect reduction declarations
  SmallVector<omp::DeclareReductionOp> reductionDecls;
  collectReductionDecls(opInst, reductionDecls);
  SmallVector<llvm::Value *> privateReductionVariables(
      opInst.getNumReductionVars());
  SmallVector<DeferredStore> deferredStores;

  auto bodyGenCB = [&](InsertPointTy allocaIP,
                       InsertPointTy codeGenIP) -> llvm::Error {
    llvm::Expected<llvm::BasicBlock *> afterAllocas = allocatePrivateVars(
        builder, moduleTranslation, privateVarsInfo, allocaIP);
    if (handleError(afterAllocas, *opInst).failed())
      return llvm::make_error<PreviouslyReportedError>();

    // Allocate reduction vars
    DenseMap<Value, llvm::Value *> reductionVariableMap;

    MutableArrayRef<BlockArgument> reductionArgs =
        cast<omp::BlockArgOpenMPOpInterface>(*opInst).getReductionBlockArgs();

    allocaIP =
        InsertPointTy(allocaIP.getBlock(),
                      allocaIP.getBlock()->getTerminator()->getIterator());

    if (failed(allocReductionVars(
            opInst, reductionArgs, builder, moduleTranslation, allocaIP,
            reductionDecls, privateReductionVariables, reductionVariableMap,
            deferredStores, isByRef)))
      return llvm::make_error<PreviouslyReportedError>();

    assert(afterAllocas.get()->getSinglePredecessor());
    builder.restoreIP(codeGenIP);

    if (handleError(
            initPrivateVars(builder, moduleTranslation, privateVarsInfo),
            *opInst)
            .failed())
      return llvm::make_error<PreviouslyReportedError>();

    if (failed(copyFirstPrivateVars(
            opInst, builder, moduleTranslation, privateVarsInfo.mlirVars,
            privateVarsInfo.llvmVars, privateVarsInfo.privatizers,
            opInst.getPrivateNeedsBarrier())))
      return llvm::make_error<PreviouslyReportedError>();

    if (failed(
            initReductionVars(opInst, reductionArgs, builder, moduleTranslation,
                              afterAllocas.get()->getSinglePredecessor(),
                              reductionDecls, privateReductionVariables,
                              reductionVariableMap, isByRef, deferredStores)))
      return llvm::make_error<PreviouslyReportedError>();

    // Save the alloca insertion point on ModuleTranslation stack for use in
    // nested regions.
    LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
        moduleTranslation, allocaIP);

    // ParallelOp has only one region associated with it.
    llvm::Expected<llvm::BasicBlock *> regionBlock = convertOmpOpRegions(
        opInst.getRegion(), "omp.par.region", builder, moduleTranslation);
    if (!regionBlock)
      return regionBlock.takeError();

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
      builder.SetInsertPoint((*regionBlock)->getTerminator());

      // Generate reductions from info
      llvm::UnreachableInst *tempTerminator = builder.CreateUnreachable();
      builder.SetInsertPoint(tempTerminator);

      llvm::OpenMPIRBuilder::InsertPointOrErrorTy contInsertPoint =
          ompBuilder->createReductions(
              builder.saveIP(), allocaIP, reductionInfos, isByRef,
              /*IsNoWait=*/false, /*IsTeamsReduction=*/false);
      if (!contInsertPoint)
        return contInsertPoint.takeError();

      if (!contInsertPoint->getBlock())
        return llvm::make_error<PreviouslyReportedError>();

      tempTerminator->eraseFromParent();
      builder.restoreIP(*contInsertPoint);
    }

    return llvm::Error::success();
  };

  auto privCB = [](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                   llvm::Value &, llvm::Value &val, llvm::Value *&replVal) {
    // tell OpenMPIRBuilder not to do anything. We handled Privatisation in
    // bodyGenCB.
    replVal = &val;
    return codeGenIP;
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) -> llvm::Error {
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
      return llvm::createStringError(
          "failed to inline `cleanup` region of `omp.declare_reduction`");

    if (failed(cleanupPrivateVars(builder, moduleTranslation, opInst.getLoc(),
                                  privateVarsInfo.llvmVars,
                                  privateVarsInfo.privatizers)))
      return llvm::make_error<PreviouslyReportedError>();

    builder.restoreIP(oldIP);
    return llvm::Error::success();
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
  bool isCancellable = constructIsCancellable(opInst);

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      ompBuilder->createParallel(ompLoc, allocaIP, bodyGenCB, privCB, finiCB,
                                 ifCond, numThreads, pbKind, isCancellable);

  if (failed(handleError(afterIP, *opInst)))
    return failure();

  builder.restoreIP(*afterIP);
  return success();
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
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  auto simdOp = cast<omp::SimdOp>(opInst);

  // TODO: Replace this with proper composite translation support.
  // Currently, simd information on composite constructs is ignored, so e.g.
  // 'do/for simd' will be treated the same as a standalone 'do/for'. This is
  // allowed by the spec, since it's equivalent to using a SIMD length of 1.
  if (simdOp.isComposite()) {
    if (failed(convertIgnoredWrapper(simdOp, moduleTranslation)))
      return failure();

    return inlineConvertOmpRegions(simdOp.getRegion(), "omp.simd.region",
                                   builder, moduleTranslation);
  }

  if (failed(checkImplementationStatus(opInst)))
    return failure();

  PrivateVarsInfo privateVarsInfo(simdOp);

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  llvm::Expected<llvm::BasicBlock *> afterAllocas = allocatePrivateVars(
      builder, moduleTranslation, privateVarsInfo, allocaIP);
  if (handleError(afterAllocas, opInst).failed())
    return failure();

  if (handleError(initPrivateVars(builder, moduleTranslation, privateVarsInfo),
                  opInst)
          .failed())
    return failure();

  llvm::ConstantInt *simdlen = nullptr;
  if (std::optional<uint64_t> simdlenVar = simdOp.getSimdlen())
    simdlen = builder.getInt64(simdlenVar.value());

  llvm::ConstantInt *safelen = nullptr;
  if (std::optional<uint64_t> safelenVar = simdOp.getSafelen())
    safelen = builder.getInt64(safelenVar.value());

  llvm::MapVector<llvm::Value *, llvm::Value *> alignedVars;
  llvm::omp::OrderKind order = convertOrderKind(simdOp.getOrder());

  llvm::BasicBlock *sourceBlock = builder.GetInsertBlock();
  std::optional<ArrayAttr> alignmentValues = simdOp.getAlignments();
  mlir::OperandRange operands = simdOp.getAlignedVars();
  for (size_t i = 0; i < operands.size(); ++i) {
    llvm::Value *alignment = nullptr;
    llvm::Value *llvmVal = moduleTranslation.lookupValue(operands[i]);
    llvm::Type *ty = llvmVal->getType();

    auto intAttr = cast<IntegerAttr>((*alignmentValues)[i]);
    alignment = builder.getInt64(intAttr.getInt());
    assert(ty->isPointerTy() && "Invalid type for aligned variable");
    assert(alignment && "Invalid alignment value");
    auto curInsert = builder.saveIP();
    builder.SetInsertPoint(sourceBlock);
    llvmVal = builder.CreateLoad(ty, llvmVal);
    builder.restoreIP(curInsert);
    alignedVars[llvmVal] = alignment;
  }

  llvm::Expected<llvm::BasicBlock *> regionBlock = convertOmpOpRegions(
      simdOp.getRegion(), "omp.simd.region", builder, moduleTranslation);

  if (failed(handleError(regionBlock, opInst)))
    return failure();

  builder.SetInsertPoint(*regionBlock, (*regionBlock)->begin());
  llvm::CanonicalLoopInfo *loopInfo = findCurrentLoopInfo(moduleTranslation);
  ompBuilder->applySimd(loopInfo, alignedVars,
                        simdOp.getIfExpr()
                            ? moduleTranslation.lookupValue(simdOp.getIfExpr())
                            : nullptr,
                        order, simdlen, safelen);

  return cleanupPrivateVars(builder, moduleTranslation, simdOp.getLoc(),
                            privateVarsInfo.llvmVars,
                            privateVarsInfo.privatizers);
}

/// Converts an OpenMP loop nest into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpLoopNest(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  auto loopOp = cast<omp::LoopNestOp>(opInst);

  // Set up the source location value for OpenMP runtime.
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  // Generator of the canonical loop body.
  SmallVector<llvm::CanonicalLoopInfo *> loopInfos;
  SmallVector<llvm::OpenMPIRBuilder::InsertPointTy> bodyInsertPoints;
  auto bodyGen = [&](llvm::OpenMPIRBuilder::InsertPointTy ip,
                     llvm::Value *iv) -> llvm::Error {
    // Make sure further conversions know about the induction variable.
    moduleTranslation.mapValue(
        loopOp.getRegion().front().getArgument(loopInfos.size()), iv);

    // Capture the body insertion point for use in nested loops. BodyIP of the
    // CanonicalLoopInfo always points to the beginning of the entry block of
    // the body.
    bodyInsertPoints.push_back(ip);

    if (loopInfos.size() != loopOp.getNumLoops() - 1)
      return llvm::Error::success();

    // Convert the body of the loop.
    builder.restoreIP(ip);
    llvm::Expected<llvm::BasicBlock *> regionBlock = convertOmpOpRegions(
        loopOp.getRegion(), "omp.loop_nest.region", builder, moduleTranslation);
    if (!regionBlock)
      return regionBlock.takeError();

    builder.SetInsertPoint(*regionBlock, (*regionBlock)->begin());
    return llvm::Error::success();
  };

  // Delegate actual loop construction to the OpenMP IRBuilder.
  // TODO: this currently assumes omp.loop_nest is semantically similar to SCF
  // loop, i.e. it has a positive step, uses signed integer semantics.
  // Reconsider this code when the nested loop operation clearly supports more
  // cases.
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

    llvm::Expected<llvm::CanonicalLoopInfo *> loopResult =
        ompBuilder->createCanonicalLoop(
            loc, bodyGen, lowerBound, upperBound, step,
            /*IsSigned=*/true, loopOp.getLoopInclusive(), computeIP);

    if (failed(handleError(loopResult, *loopOp)))
      return failure();

    loopInfos.push_back(*loopResult);
  }

  // Collapse loops. Store the insertion point because LoopInfos may get
  // invalidated.
  llvm::OpenMPIRBuilder::InsertPointTy afterIP =
      loopInfos.front()->getAfterIP();

  // Update the stack frame created for this loop to point to the resulting loop
  // after applying transformations.
  moduleTranslation.stackWalk<OpenMPLoopInfoStackFrame>(
      [&](OpenMPLoopInfoStackFrame &frame) {
        frame.loopInfo = ompBuilder->collapseLoops(ompLoc.DL, loopInfos, {});
        return WalkResult::interrupt();
      });

  // Continue building IR after the loop. Note that the LoopInfo returned by
  // `collapseLoops` points inside the outermost loop and is intended for
  // potential further loop transformations. Use the insertion point stored
  // before collapsing loops instead.
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
  if (failed(checkImplementationStatus(opInst)))
    return failure();

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  llvm::AtomicOrdering AO = convertAtomicOrdering(readOp.getMemoryOrder());
  llvm::Value *x = moduleTranslation.lookupValue(readOp.getX());
  llvm::Value *v = moduleTranslation.lookupValue(readOp.getV());

  llvm::Type *elementType =
      moduleTranslation.convertType(readOp.getElementType());

  llvm::OpenMPIRBuilder::AtomicOpValue V = {v, elementType, false, false};
  llvm::OpenMPIRBuilder::AtomicOpValue X = {x, elementType, false, false};
  builder.restoreIP(ompBuilder->createAtomicRead(ompLoc, X, V, AO, allocaIP));
  return success();
}

/// Converts an omp.atomic.write operation to LLVM IR.
static LogicalResult
convertOmpAtomicWrite(Operation &opInst, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  auto writeOp = cast<omp::AtomicWriteOp>(opInst);
  if (failed(checkImplementationStatus(opInst)))
    return failure();

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::AtomicOrdering ao = convertAtomicOrdering(writeOp.getMemoryOrder());
  llvm::Value *expr = moduleTranslation.lookupValue(writeOp.getExpr());
  llvm::Value *dest = moduleTranslation.lookupValue(writeOp.getX());
  llvm::Type *ty = moduleTranslation.convertType(writeOp.getExpr().getType());
  llvm::OpenMPIRBuilder::AtomicOpValue x = {dest, ty, /*isSigned=*/false,
                                            /*isVolatile=*/false};
  builder.restoreIP(
      ompBuilder->createAtomicWrite(ompLoc, x, expr, ao, allocaIP));
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
  if (failed(checkImplementationStatus(*opInst)))
    return failure();

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
  auto updateFn =
      [&opInst, &moduleTranslation](
          llvm::Value *atomicx,
          llvm::IRBuilder<> &builder) -> llvm::Expected<llvm::Value *> {
    Block &bb = *opInst.getRegion().begin();
    moduleTranslation.mapValue(*opInst.getRegion().args_begin(), atomicx);
    moduleTranslation.mapBlock(&bb, builder.GetInsertBlock());
    if (failed(moduleTranslation.convertBlock(bb, true, builder)))
      return llvm::make_error<PreviouslyReportedError>();

    omp::YieldOp yieldop = dyn_cast<omp::YieldOp>(bb.getTerminator());
    assert(yieldop && yieldop.getResults().size() == 1 &&
           "terminator must be omp.yield op and it must have exactly one "
           "argument");
    return moduleTranslation.lookupValue(yieldop.getResults()[0]);
  };

  // Handle ambiguous alloca, if any.
  auto allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      ompBuilder->createAtomicUpdate(ompLoc, allocaIP, llvmAtomicX, llvmExpr,
                                     atomicOrdering, binop, updateFn,
                                     isXBinopExpr);

  if (failed(handleError(afterIP, *opInst)))
    return failure();

  builder.restoreIP(*afterIP);
  return success();
}

static LogicalResult
convertOmpAtomicCapture(omp::AtomicCaptureOp atomicCaptureOp,
                        llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  if (failed(checkImplementationStatus(*atomicCaptureOp)))
    return failure();

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
    // Find the binary update operation that uses the region argument
    // and get the expression to update
    if (innerOpList.size() == 2) {
      mlir::Operation &innerOp = *atomicUpdateOp.getRegion().front().begin();
      if (!llvm::is_contained(innerOp.getOperands(),
                              atomicUpdateOp.getRegion().getArgument(0))) {
        return atomicUpdateOp.emitError(
            "no atomic update operation with region argument"
            " as operand found inside atomic.update region");
      }
      binop = convertBinOpToAtomic(innerOp);
      isXBinopExpr =
          innerOp.getOperand(0) == atomicUpdateOp.getRegion().getArgument(0);
      mlirExpr = (isXBinopExpr ? innerOp.getOperand(1) : innerOp.getOperand(0));
    } else {
      binop = llvm::AtomicRMWInst::BinOp::BAD_BINOP;
    }
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

  auto updateFn =
      [&](llvm::Value *atomicx,
          llvm::IRBuilder<> &builder) -> llvm::Expected<llvm::Value *> {
    if (atomicWriteOp)
      return moduleTranslation.lookupValue(atomicWriteOp.getExpr());
    Block &bb = *atomicUpdateOp.getRegion().begin();
    moduleTranslation.mapValue(*atomicUpdateOp.getRegion().args_begin(),
                               atomicx);
    moduleTranslation.mapBlock(&bb, builder.GetInsertBlock());
    if (failed(moduleTranslation.convertBlock(bb, true, builder)))
      return llvm::make_error<PreviouslyReportedError>();

    omp::YieldOp yieldop = dyn_cast<omp::YieldOp>(bb.getTerminator());
    assert(yieldop && yieldop.getResults().size() == 1 &&
           "terminator must be omp.yield op and it must have exactly one "
           "argument");
    return moduleTranslation.lookupValue(yieldop.getResults()[0]);
  };

  // Handle ambiguous alloca, if any.
  auto allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      ompBuilder->createAtomicCapture(
          ompLoc, allocaIP, llvmAtomicX, llvmAtomicV, llvmExpr, atomicOrdering,
          binop, updateFn, atomicUpdateOp, isPostfixUpdate, isXBinopExpr);

  if (failed(handleError(afterIP, *atomicCaptureOp)))
    return failure();

  builder.restoreIP(*afterIP);
  return success();
}

static llvm::omp::Directive convertCancellationConstructType(
    omp::ClauseCancellationConstructType directive) {
  switch (directive) {
  case omp::ClauseCancellationConstructType::Loop:
    return llvm::omp::Directive::OMPD_for;
  case omp::ClauseCancellationConstructType::Parallel:
    return llvm::omp::Directive::OMPD_parallel;
  case omp::ClauseCancellationConstructType::Sections:
    return llvm::omp::Directive::OMPD_sections;
  case omp::ClauseCancellationConstructType::Taskgroup:
    return llvm::omp::Directive::OMPD_taskgroup;
  }
}

static LogicalResult
convertOmpCancel(omp::CancelOp op, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  if (failed(checkImplementationStatus(*op.getOperation())))
    return failure();

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  llvm::Value *ifCond = nullptr;
  if (Value ifVar = op.getIfExpr())
    ifCond = moduleTranslation.lookupValue(ifVar);

  llvm::omp::Directive cancelledDirective =
      convertCancellationConstructType(op.getCancelDirective());

  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      ompBuilder->createCancel(ompLoc, ifCond, cancelledDirective);

  if (failed(handleError(afterIP, *op.getOperation())))
    return failure();

  builder.restoreIP(afterIP.get());

  return success();
}

static LogicalResult
convertOmpCancellationPoint(omp::CancellationPointOp op,
                            llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation) {
  if (failed(checkImplementationStatus(*op.getOperation())))
    return failure();

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  llvm::omp::Directive cancelledDirective =
      convertCancellationConstructType(op.getCancelDirective());

  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      ompBuilder->createCancellationPoint(ompLoc, cancelledDirective);

  if (failed(handleError(afterIP, *op.getOperation())))
    return failure();

  builder.restoreIP(afterIP.get());

  return success();
}

/// Converts an OpenMP Threadprivate operation into LLVM IR using
/// OpenMPIRBuilder.
static LogicalResult
convertOmpThreadprivate(Operation &opInst, llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  auto threadprivateOp = cast<omp::ThreadprivateOp>(opInst);

  if (failed(checkImplementationStatus(opInst)))
    return failure();

  Value symAddr = threadprivateOp.getSymAddr();
  auto *symOp = symAddr.getDefiningOp();

  if (auto asCast = dyn_cast<LLVM::AddrSpaceCastOp>(symOp))
    symOp = asCast.getOperand().getDefiningOp();

  if (!isa<LLVM::AddressOfOp>(symOp))
    return opInst.emitError("Addressing symbol not found");
  LLVM::AddressOfOp addressOfOp = dyn_cast<LLVM::AddressOfOp>(symOp);

  LLVM::GlobalOp global =
      addressOfOp.getGlobal(moduleTranslation.symbolTable());
  llvm::GlobalValue *globalValue = moduleTranslation.lookupGlobal(global);

  if (!ompBuilder->Config.isTargetDevice()) {
    llvm::Type *type = globalValue->getValueType();
    llvm::TypeSize typeSize =
        builder.GetInsertBlock()->getModule()->getDataLayout().getTypeStoreSize(
            type);
    llvm::ConstantInt *size = builder.getInt64(typeSize.getFixedValue());
    llvm::Value *callInst = ompBuilder->createCachedThreadPrivate(
        ompLoc, globalValue, size, global.getSymName() + ".cache");
    moduleTranslation.mapValue(opInst.getResult(0), callInst);
  } else {
    moduleTranslation.mapValue(opInst.getResult(0), globalValue);
  }

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
// Append customMappers information to existing MapInfosTy
struct MapInfosTy : llvm::OpenMPIRBuilder::MapInfosTy {
  SmallVector<Operation *, 4> Mappers;

  /// Append arrays in \a CurInfo.
  void append(MapInfosTy &curInfo) {
    Mappers.append(curInfo.Mappers.begin(), curInfo.Mappers.end());
    llvm::OpenMPIRBuilder::MapInfosTy::append(curInfo);
  }
};
// A small helper structure to contain data gathered
// for map lowering and coalese it into one area and
// avoiding extra computations such as searches in the
// llvm module for lowered mapped variables or checking
// if something is declare target (and retrieving the
// value) more than neccessary.
struct MapInfoData : MapInfosTy {
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
    MapInfosTy::append(CurInfo);
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
    llvm::IRBuilderBase &builder, ArrayRef<Value> useDevPtrOperands = {},
    ArrayRef<Value> useDevAddrOperands = {},
    ArrayRef<Value> hasDevAddrOperands = {}) {
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
        llvm::omp::OpenMPOffloadMappingFlags(mapOp.getMapType()));
    mapData.Names.push_back(LLVM::createMappingInformation(
        mapOp.getLoc(), *moduleTranslation.getOpenMPBuilder()));
    mapData.DevicePointers.push_back(llvm::OpenMPIRBuilder::DeviceInfoTy::None);
    if (mapOp.getMapperId())
      mapData.Mappers.push_back(
          SymbolTable::lookupNearestSymbolFrom<omp::DeclareMapperOp>(
              mapOp, mapOp.getMapperIdAttr()));
    else
      mapData.Mappers.push_back(nullptr);
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
        mapData.Mappers.push_back(nullptr);
        mapData.IsAMapping.push_back(false);
        mapData.IsAMember.push_back(checkIsAMember(useDevOperands, mapOp));
      }
    }
  };

  addDevInfos(useDevAddrOperands, llvm::OpenMPIRBuilder::DeviceInfoTy::Address);
  addDevInfos(useDevPtrOperands, llvm::OpenMPIRBuilder::DeviceInfoTy::Pointer);

  for (Value mapValue : hasDevAddrOperands) {
    auto mapOp = cast<omp::MapInfoOp>(mapValue.getDefiningOp());
    Value offloadPtr =
        mapOp.getVarPtrPtr() ? mapOp.getVarPtrPtr() : mapOp.getVarPtr();
    llvm::Value *origValue = moduleTranslation.lookupValue(offloadPtr);
    auto mapType =
        static_cast<llvm::omp::OpenMPOffloadMappingFlags>(mapOp.getMapType());
    auto mapTypeAlways = llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_ALWAYS;

    mapData.OriginalValue.push_back(origValue);
    mapData.BasePointers.push_back(origValue);
    mapData.Pointers.push_back(origValue);
    mapData.IsDeclareTarget.push_back(false);
    mapData.BaseType.push_back(
        moduleTranslation.convertType(mapOp.getVarType()));
    mapData.Sizes.push_back(
        builder.getInt64(dl.getTypeSize(mapOp.getVarType())));
    mapData.MapClause.push_back(mapOp.getOperation());
    if (llvm::to_underlying(mapType & mapTypeAlways)) {
      // Descriptors are mapped with the ALWAYS flag, since they can get
      // rematerialized, so the address of the decriptor for a given object
      // may change from one place to another.
      mapData.Types.push_back(mapType);
      // Technically it's possible for a non-descriptor mapping to have
      // both has-device-addr and ALWAYS, so lookup the mapper in case it
      // exists.
      if (mapOp.getMapperId()) {
        mapData.Mappers.push_back(
            SymbolTable::lookupNearestSymbolFrom<omp::DeclareMapperOp>(
                mapOp, mapOp.getMapperIdAttr()));
      } else {
        mapData.Mappers.push_back(nullptr);
      }
    } else {
      mapData.Types.push_back(
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_LITERAL);
      mapData.Mappers.push_back(nullptr);
    }
    mapData.Names.push_back(LLVM::createMappingInformation(
        mapOp.getLoc(), *moduleTranslation.getOpenMPBuilder()));
    mapData.DevicePointers.push_back(
        llvm::OpenMPIRBuilder::DeviceInfoTy::Address);
    mapData.IsAMapping.push_back(false);
    mapData.IsAMember.push_back(checkIsAMember(hasDevAddrOperands, mapOp));
  }
}

static int getMapDataMemberIdx(MapInfoData &mapData, omp::MapInfoOp memberOp) {
  auto *res = llvm::find(mapData.MapClause, memberOp);
  assert(res != mapData.MapClause.end() &&
         "MapInfoOp for member not found in MapData, cannot return index");
  return std::distance(mapData.MapClause.begin(), res);
}

static omp::MapInfoOp getFirstOrLastMappedMemberPtr(omp::MapInfoOp mapInfo,
                                                    bool first) {
  ArrayAttr indexAttr = mapInfo.getMembersIndexAttr();
  // Only 1 member has been mapped, we can return it.
  if (indexAttr.size() == 1)
    return cast<omp::MapInfoOp>(mapInfo.getMembers()[0].getDefiningOp());

  llvm::SmallVector<size_t> indices(indexAttr.size());
  std::iota(indices.begin(), indices.end(), 0);

  llvm::sort(indices.begin(), indices.end(),
             [&](const size_t a, const size_t b) {
               auto memberIndicesA = cast<ArrayAttr>(indexAttr[a]);
               auto memberIndicesB = cast<ArrayAttr>(indexAttr[b]);
               for (const auto it : llvm::zip(memberIndicesA, memberIndicesB)) {
                 int64_t aIndex = cast<IntegerAttr>(std::get<0>(it)).getInt();
                 int64_t bIndex = cast<IntegerAttr>(std::get<1>(it)).getInt();

                 if (aIndex == bIndex)
                   continue;

                 if (aIndex < bIndex)
                   return first;

                 if (aIndex > bIndex)
                   return !first;
               }

               // Iterated the up until the end of the smallest member and
               // they were found to be equal up to that point, so select
               // the member with the lowest index count, so the "parent"
               return memberIndicesA.size() < memberIndicesB.size();
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
    llvm::OpenMPIRBuilder &ompBuilder, DataLayout &dl, MapInfosTy &combinedInfo,
    MapInfoData &mapData, uint64_t mapDataIndex, bool isTargetParams) {
  assert(!ompBuilder.Config.isTargetDevice() &&
         "function only supported for host device codegen");

  // Map the first segment of our structure
  combinedInfo.Types.emplace_back(
      isTargetParams
          ? llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TARGET_PARAM
          : llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE);
  combinedInfo.DevicePointers.emplace_back(
      mapData.DevicePointers[mapDataIndex]);
  combinedInfo.Mappers.emplace_back(mapData.Mappers[mapDataIndex]);
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

  llvm::omp::OpenMPOffloadMappingFlags memberOfFlag =
      ompBuilder.getMemberOfFlag(combinedInfo.BasePointers.size() - 1);

  // This creates the initial MEMBER_OF mapping that consists of
  // the parent/top level container (same as above effectively, except
  // with a fixed initial compile time size and separate maptype which
  // indicates the true mape type (tofrom etc.). This parent mapping is
  // only relevant if the structure in its totality is being mapped,
  // otherwise the above suffices.
  if (!parentClause.getPartialMap()) {
    // TODO: This will need to be expanded to include the whole host of logic
    // for the map flags that Clang currently supports (e.g. it should do some
    // further case specific flag modifications). For the moment, it handles
    // what we support as expected.
    llvm::omp::OpenMPOffloadMappingFlags mapFlag = mapData.Types[mapDataIndex];
    ompBuilder.setCorrectMemberOfFlag(mapFlag, memberOfFlag);
    combinedInfo.Types.emplace_back(mapFlag);
    combinedInfo.DevicePointers.emplace_back(
        llvm::OpenMPIRBuilder::DeviceInfoTy::None);
    combinedInfo.Mappers.emplace_back(nullptr);
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
    llvm::OpenMPIRBuilder &ompBuilder, DataLayout &dl, MapInfosTy &combinedInfo,
    MapInfoData &mapData, uint64_t mapDataIndex,
    llvm::omp::OpenMPOffloadMappingFlags memberOfFlag) {
  assert(!ompBuilder.Config.isTargetDevice() &&
         "function only supported for host device codegen");

  auto parentClause =
      llvm::cast<omp::MapInfoOp>(mapData.MapClause[mapDataIndex]);

  for (auto mappedMembers : parentClause.getMembers()) {
    auto memberClause =
        llvm::cast<omp::MapInfoOp>(mappedMembers.getDefiningOp());
    int memberDataIdx = getMapDataMemberIdx(mapData, memberClause);

    assert(memberDataIdx >= 0 && "could not find mapped member of structure");

    // If we're currently mapping a pointer to a block of data, we must
    // initially map the pointer, and then attatch/bind the data with a
    // subsequent map to the pointer. This segment of code generates the
    // pointer mapping, which can in certain cases be optimised out as Clang
    // currently does in its lowering. However, for the moment we do not do so,
    // in part as we currently have substantially less information on the data
    // being mapped at this stage.
    if (checkIfPointerMap(memberClause)) {
      auto mapFlag =
          llvm::omp::OpenMPOffloadMappingFlags(memberClause.getMapType());
      mapFlag &= ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TARGET_PARAM;
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_MEMBER_OF;
      ompBuilder.setCorrectMemberOfFlag(mapFlag, memberOfFlag);
      combinedInfo.Types.emplace_back(mapFlag);
      combinedInfo.DevicePointers.emplace_back(
          llvm::OpenMPIRBuilder::DeviceInfoTy::None);
      combinedInfo.Mappers.emplace_back(nullptr);
      combinedInfo.Names.emplace_back(
          LLVM::createMappingInformation(memberClause.getLoc(), ompBuilder));
      combinedInfo.BasePointers.emplace_back(
          mapData.BasePointers[mapDataIndex]);
      combinedInfo.Pointers.emplace_back(mapData.BasePointers[memberDataIdx]);
      combinedInfo.Sizes.emplace_back(builder.getInt64(
          moduleTranslation.getLLVMModule()->getDataLayout().getPointerSize()));
    }

    // Same MemberOfFlag to indicate its link with parent and other members
    // of.
    auto mapFlag =
        llvm::omp::OpenMPOffloadMappingFlags(memberClause.getMapType());
    mapFlag &= ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TARGET_PARAM;
    mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_MEMBER_OF;
    ompBuilder.setCorrectMemberOfFlag(mapFlag, memberOfFlag);
    if (checkIfPointerMap(memberClause))
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PTR_AND_OBJ;

    combinedInfo.Types.emplace_back(mapFlag);
    combinedInfo.DevicePointers.emplace_back(
        mapData.DevicePointers[memberDataIdx]);
    combinedInfo.Mappers.emplace_back(mapData.Mappers[memberDataIdx]);
    combinedInfo.Names.emplace_back(
        LLVM::createMappingInformation(memberClause.getLoc(), ompBuilder));
    uint64_t basePointerIndex =
        checkIfPointerMap(memberClause) ? memberDataIdx : mapDataIndex;
    combinedInfo.BasePointers.emplace_back(
        mapData.BasePointers[basePointerIndex]);
    combinedInfo.Pointers.emplace_back(mapData.Pointers[memberDataIdx]);

    llvm::Value *size = mapData.Sizes[memberDataIdx];
    if (checkIfPointerMap(memberClause)) {
      size = builder.CreateSelect(
          builder.CreateIsNull(mapData.Pointers[memberDataIdx]),
          builder.getInt64(0), size);
    }

    combinedInfo.Sizes.emplace_back(size);
  }
}

static void processIndividualMap(MapInfoData &mapData, size_t mapDataIdx,
                                 MapInfosTy &combinedInfo, bool isTargetParams,
                                 int mapDataParentIdx = -1) {
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

  if (mapInfoOp.getMapCaptureType() == omp::VariableCaptureKind::ByCopy &&
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
  combinedInfo.Mappers.emplace_back(mapData.Mappers[mapDataIdx]);
  combinedInfo.Names.emplace_back(mapData.Names[mapDataIdx]);
  combinedInfo.Types.emplace_back(mapFlag);
  combinedInfo.Sizes.emplace_back(mapData.Sizes[mapDataIdx]);
}

static void processMapWithMembersOf(LLVM::ModuleTranslation &moduleTranslation,
                                    llvm::IRBuilderBase &builder,
                                    llvm::OpenMPIRBuilder &ompBuilder,
                                    DataLayout &dl, MapInfosTy &combinedInfo,
                                    MapInfoData &mapData, uint64_t mapDataIndex,
                                    bool isTargetParams) {
  assert(!ompBuilder.Config.isTargetDevice() &&
         "function only supported for host device codegen");

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
  assert(!moduleTranslation.getOpenMPBuilder()->Config.isTargetDevice() &&
         "function only supported for host device codegen");
  for (size_t i = 0; i < mapData.MapClause.size(); ++i) {
    // if it's declare target, skip it, it's handled separately.
    if (!mapData.IsDeclareTarget[i]) {
      auto mapOp = cast<omp::MapInfoOp>(mapData.MapClause[i]);
      omp::VariableCaptureKind captureKind = mapOp.getMapCaptureType();
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
                        DataLayout &dl, MapInfosTy &combinedInfo,
                        MapInfoData &mapData, bool isTargetParams = false) {
  assert(!moduleTranslation.getOpenMPBuilder()->Config.isTargetDevice() &&
         "function only supported for host device codegen");

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

static llvm::Expected<llvm::Function *>
emitUserDefinedMapper(Operation *declMapperOp, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation,
                      llvm::StringRef mapperFuncName);

static llvm::Expected<llvm::Function *>
getOrCreateUserDefinedMapperFunc(Operation *op, llvm::IRBuilderBase &builder,
                                 LLVM::ModuleTranslation &moduleTranslation) {
  assert(!moduleTranslation.getOpenMPBuilder()->Config.isTargetDevice() &&
         "function only supported for host device codegen");
  auto declMapperOp = cast<omp::DeclareMapperOp>(op);
  std::string mapperFuncName =
      moduleTranslation.getOpenMPBuilder()->createPlatformSpecificName(
          {"omp_mapper", declMapperOp.getSymName()});

  if (auto *lookupFunc = moduleTranslation.lookupFunction(mapperFuncName))
    return lookupFunc;

  return emitUserDefinedMapper(declMapperOp, builder, moduleTranslation,
                               mapperFuncName);
}

static llvm::Expected<llvm::Function *>
emitUserDefinedMapper(Operation *op, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation,
                      llvm::StringRef mapperFuncName) {
  assert(!moduleTranslation.getOpenMPBuilder()->Config.isTargetDevice() &&
         "function only supported for host device codegen");
  auto declMapperOp = cast<omp::DeclareMapperOp>(op);
  auto declMapperInfoOp = declMapperOp.getDeclareMapperInfo();
  DataLayout dl = DataLayout(declMapperOp->getParentOfType<ModuleOp>());
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Type *varType = moduleTranslation.convertType(declMapperOp.getType());
  SmallVector<Value> mapVars = declMapperInfoOp.getMapVars();

  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;

  // Fill up the arrays with all the mapped variables.
  MapInfosTy combinedInfo;
  auto genMapInfoCB =
      [&](InsertPointTy codeGenIP, llvm::Value *ptrPHI,
          llvm::Value *unused2) -> llvm::OpenMPIRBuilder::MapInfosOrErrorTy {
    builder.restoreIP(codeGenIP);
    moduleTranslation.mapValue(declMapperOp.getSymVal(), ptrPHI);
    moduleTranslation.mapBlock(&declMapperOp.getRegion().front(),
                               builder.GetInsertBlock());
    if (failed(moduleTranslation.convertBlock(declMapperOp.getRegion().front(),
                                              /*ignoreArguments=*/true,
                                              builder)))
      return llvm::make_error<PreviouslyReportedError>();
    MapInfoData mapData;
    collectMapDataFromMapOperands(mapData, mapVars, moduleTranslation, dl,
                                  builder);
    genMapInfos(builder, moduleTranslation, dl, combinedInfo, mapData);

    // Drop the mapping that is no longer necessary so that the same region can
    // be processed multiple times.
    moduleTranslation.forgetMapping(declMapperOp.getRegion());
    return combinedInfo;
  };

  auto customMapperCB = [&](unsigned i) -> llvm::Expected<llvm::Function *> {
    if (!combinedInfo.Mappers[i])
      return nullptr;
    return getOrCreateUserDefinedMapperFunc(combinedInfo.Mappers[i], builder,
                                            moduleTranslation);
  };

  llvm::Expected<llvm::Function *> newFn = ompBuilder->emitUserDefinedMapper(
      genMapInfoCB, varType, mapperFuncName, customMapperCB);
  if (!newFn)
    return newFn.takeError();
  moduleTranslation.mapFunction(mapperFuncName, *newFn);
  return *newFn;
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
  llvm::OpenMPIRBuilder::TargetDataInfo info(/*RequiresDevicePointerInfo=*/true,
                                             /*SeparateBeginEndCalls=*/true);

  LogicalResult result =
      llvm::TypeSwitch<Operation *, LogicalResult>(op)
          .Case([&](omp::TargetDataOp dataOp) {
            if (failed(checkImplementationStatus(*dataOp)))
              return failure();

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
          .Case([&](omp::TargetEnterDataOp enterDataOp) -> LogicalResult {
            if (failed(checkImplementationStatus(*enterDataOp)))
              return failure();

            if (auto ifVar = enterDataOp.getIfExpr())
              ifCond = moduleTranslation.lookupValue(ifVar);

            if (auto devId = enterDataOp.getDevice())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(devId.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  deviceID = intAttr.getInt();
            RTLFn =
                enterDataOp.getNowait()
                    ? llvm::omp::OMPRTL___tgt_target_data_begin_nowait_mapper
                    : llvm::omp::OMPRTL___tgt_target_data_begin_mapper;
            mapVars = enterDataOp.getMapVars();
            info.HasNoWait = enterDataOp.getNowait();
            return success();
          })
          .Case([&](omp::TargetExitDataOp exitDataOp) -> LogicalResult {
            if (failed(checkImplementationStatus(*exitDataOp)))
              return failure();

            if (auto ifVar = exitDataOp.getIfExpr())
              ifCond = moduleTranslation.lookupValue(ifVar);

            if (auto devId = exitDataOp.getDevice())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(devId.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  deviceID = intAttr.getInt();

            RTLFn = exitDataOp.getNowait()
                        ? llvm::omp::OMPRTL___tgt_target_data_end_nowait_mapper
                        : llvm::omp::OMPRTL___tgt_target_data_end_mapper;
            mapVars = exitDataOp.getMapVars();
            info.HasNoWait = exitDataOp.getNowait();
            return success();
          })
          .Case([&](omp::TargetUpdateOp updateDataOp) -> LogicalResult {
            if (failed(checkImplementationStatus(*updateDataOp)))
              return failure();

            if (auto ifVar = updateDataOp.getIfExpr())
              ifCond = moduleTranslation.lookupValue(ifVar);

            if (auto devId = updateDataOp.getDevice())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(devId.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  deviceID = intAttr.getInt();

            RTLFn =
                updateDataOp.getNowait()
                    ? llvm::omp::OMPRTL___tgt_target_data_update_nowait_mapper
                    : llvm::omp::OMPRTL___tgt_target_data_update_mapper;
            mapVars = updateDataOp.getMapVars();
            info.HasNoWait = updateDataOp.getNowait();
            return success();
          })
          .Default([&](Operation *op) {
            llvm_unreachable("unexpected operation");
            return failure();
          });

  if (failed(result))
    return failure();

  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  MapInfoData mapData;
  collectMapDataFromMapOperands(mapData, mapVars, moduleTranslation, DL,
                                builder, useDevicePtrVars, useDeviceAddrVars);

  // Fill up the arrays with all the mapped variables.
  MapInfosTy combinedInfo;
  auto genMapInfoCB = [&](InsertPointTy codeGenIP) -> MapInfosTy & {
    builder.restoreIP(codeGenIP);
    genMapInfos(builder, moduleTranslation, DL, combinedInfo, mapData);
    return combinedInfo;
  };

  // Define a lambda to apply mappings between use_device_addr and
  // use_device_ptr base pointers, and their associated block arguments.
  auto mapUseDevice =
      [&moduleTranslation](
          llvm::OpenMPIRBuilder::DeviceInfoTy type,
          llvm::ArrayRef<BlockArgument> blockArgs,
          llvm::SmallVectorImpl<Value> &useDeviceVars, MapInfoData &mapInfoData,
          llvm::function_ref<llvm::Value *(llvm::Value *)> mapper = nullptr) {
        for (auto [arg, useDevVar] :
             llvm::zip_equal(blockArgs, useDeviceVars)) {

          auto getMapBasePtr = [](omp::MapInfoOp mapInfoOp) {
            return mapInfoOp.getVarPtrPtr() ? mapInfoOp.getVarPtrPtr()
                                            : mapInfoOp.getVarPtr();
          };

          auto useDevMap = cast<omp::MapInfoOp>(useDevVar.getDefiningOp());
          for (auto [mapClause, devicePointer, basePointer] : llvm::zip_equal(
                   mapInfoData.MapClause, mapInfoData.DevicePointers,
                   mapInfoData.BasePointers)) {
            auto mapOp = cast<omp::MapInfoOp>(mapClause);
            if (getMapBasePtr(mapOp) != getMapBasePtr(useDevMap) ||
                devicePointer != type)
              continue;

            if (llvm::Value *devPtrInfoMap =
                    mapper ? mapper(basePointer) : basePointer) {
              moduleTranslation.mapValue(arg, devPtrInfoMap);
              break;
            }
          }
        }
      };

  using BodyGenTy = llvm::OpenMPIRBuilder::BodyGenTy;
  auto bodyGenCB = [&](InsertPointTy codeGenIP, BodyGenTy bodyGenType)
      -> llvm::OpenMPIRBuilder::InsertPointOrErrorTy {
    builder.restoreIP(codeGenIP);
    assert(isa<omp::TargetDataOp>(op) &&
           "BodyGen requested for non TargetDataOp");
    auto blockArgIface = cast<omp::BlockArgOpenMPOpInterface>(op);
    Region &region = cast<omp::TargetDataOp>(op).getRegion();
    switch (bodyGenType) {
    case BodyGenTy::Priv:
      // Check if any device ptr/addr info is available
      if (!info.DevicePtrInfoMap.empty()) {
        mapUseDevice(llvm::OpenMPIRBuilder::DeviceInfoTy::Address,
                     blockArgIface.getUseDeviceAddrBlockArgs(),
                     useDeviceAddrVars, mapData,
                     [&](llvm::Value *basePointer) -> llvm::Value * {
                       if (!info.DevicePtrInfoMap[basePointer].second)
                         return nullptr;
                       return builder.CreateLoad(
                           builder.getPtrTy(),
                           info.DevicePtrInfoMap[basePointer].second);
                     });
        mapUseDevice(llvm::OpenMPIRBuilder::DeviceInfoTy::Pointer,
                     blockArgIface.getUseDevicePtrBlockArgs(), useDevicePtrVars,
                     mapData, [&](llvm::Value *basePointer) {
                       return info.DevicePtrInfoMap[basePointer].second;
                     });

        if (failed(inlineConvertOmpRegions(region, "omp.data.region", builder,
                                           moduleTranslation)))
          return llvm::make_error<PreviouslyReportedError>();
      }
      break;
    case BodyGenTy::DupNoPriv:
      // We must always restoreIP regardless of doing anything the caller
      // does not restore it, leading to incorrect (no) branch generation.
      builder.restoreIP(codeGenIP);
      break;
    case BodyGenTy::NoPriv:
      // If device info is available then region has already been generated
      if (info.DevicePtrInfoMap.empty()) {
        // For device pass, if use_device_ptr(addr) mappings were present,
        // we need to link them here before codegen.
        if (ompBuilder->Config.IsTargetDevice.value_or(false)) {
          mapUseDevice(llvm::OpenMPIRBuilder::DeviceInfoTy::Address,
                       blockArgIface.getUseDeviceAddrBlockArgs(),
                       useDeviceAddrVars, mapData);
          mapUseDevice(llvm::OpenMPIRBuilder::DeviceInfoTy::Pointer,
                       blockArgIface.getUseDevicePtrBlockArgs(),
                       useDevicePtrVars, mapData);
        }

        if (failed(inlineConvertOmpRegions(region, "omp.data.region", builder,
                                           moduleTranslation)))
          return llvm::make_error<PreviouslyReportedError>();
      }
      break;
    }
    return builder.saveIP();
  };

  auto customMapperCB =
      [&](unsigned int i) -> llvm::Expected<llvm::Function *> {
    if (!combinedInfo.Mappers[i])
      return nullptr;
    info.HasMapper = true;
    return getOrCreateUserDefinedMapperFunc(combinedInfo.Mappers[i], builder,
                                            moduleTranslation);
  };

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP = [&]() {
    if (isa<omp::TargetDataOp>(op))
      return ompBuilder->createTargetData(ompLoc, allocaIP, builder.saveIP(),
                                          builder.getInt64(deviceID), ifCond,
                                          info, genMapInfoCB, customMapperCB,
                                          /*MapperFunc=*/nullptr, bodyGenCB,
                                          /*DeviceAddrCB=*/nullptr);
    return ompBuilder->createTargetData(
        ompLoc, allocaIP, builder.saveIP(), builder.getInt64(deviceID), ifCond,
        info, genMapInfoCB, customMapperCB, &RTLFn);
  }();

  if (failed(handleError(afterIP, *op)))
    return failure();

  builder.restoreIP(*afterIP);
  return success();
}

static LogicalResult
convertOmpDistribute(Operation &opInst, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  auto distributeOp = cast<omp::DistributeOp>(opInst);
  if (failed(checkImplementationStatus(opInst)))
    return failure();

  /// Process teams op reduction in distribute if the reduction is contained in
  /// the distribute op.
  omp::TeamsOp teamsOp = opInst.getParentOfType<omp::TeamsOp>();
  bool doDistributeReduction =
      teamsOp ? teamsReductionContainedInDistribute(teamsOp) : false;

  DenseMap<Value, llvm::Value *> reductionVariableMap;
  unsigned numReductionVars = teamsOp ? teamsOp.getNumReductionVars() : 0;
  SmallVector<omp::DeclareReductionOp> reductionDecls;
  SmallVector<llvm::Value *> privateReductionVariables(numReductionVars);
  llvm::ArrayRef<bool> isByRef;

  if (doDistributeReduction) {
    isByRef = getIsByRef(teamsOp.getReductionByref());
    assert(isByRef.size() == teamsOp.getNumReductionVars());

    collectReductionDecls(teamsOp, reductionDecls);
    llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
        findAllocaInsertPoint(builder, moduleTranslation);

    MutableArrayRef<BlockArgument> reductionArgs =
        llvm::cast<omp::BlockArgOpenMPOpInterface>(*teamsOp)
            .getReductionBlockArgs();

    if (failed(allocAndInitializeReductionVars(
            teamsOp, reductionArgs, builder, moduleTranslation, allocaIP,
            reductionDecls, privateReductionVariables, reductionVariableMap,
            isByRef)))
      return failure();
  }

  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto bodyGenCB = [&](InsertPointTy allocaIP,
                       InsertPointTy codeGenIP) -> llvm::Error {
    // Save the alloca insertion point on ModuleTranslation stack for use in
    // nested regions.
    LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
        moduleTranslation, allocaIP);

    // DistributeOp has only one region associated with it.
    builder.restoreIP(codeGenIP);
    PrivateVarsInfo privVarsInfo(distributeOp);

    llvm::Expected<llvm::BasicBlock *> afterAllocas =
        allocatePrivateVars(builder, moduleTranslation, privVarsInfo, allocaIP);
    if (handleError(afterAllocas, opInst).failed())
      return llvm::make_error<PreviouslyReportedError>();

    if (handleError(initPrivateVars(builder, moduleTranslation, privVarsInfo),
                    opInst)
            .failed())
      return llvm::make_error<PreviouslyReportedError>();

    if (failed(copyFirstPrivateVars(
            distributeOp, builder, moduleTranslation, privVarsInfo.mlirVars,
            privVarsInfo.llvmVars, privVarsInfo.privatizers,
            distributeOp.getPrivateNeedsBarrier())))
      return llvm::make_error<PreviouslyReportedError>();

    llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
    llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
    llvm::Expected<llvm::BasicBlock *> regionBlock =
        convertOmpOpRegions(distributeOp.getRegion(), "omp.distribute.region",
                            builder, moduleTranslation);
    if (!regionBlock)
      return regionBlock.takeError();
    builder.SetInsertPoint(*regionBlock, (*regionBlock)->begin());

    // Skip applying a workshare loop below when translating 'distribute
    // parallel do' (it's been already handled by this point while translating
    // the nested omp.wsloop).
    if (!isa_and_present<omp::WsloopOp>(distributeOp.getNestedWrapper())) {
      // TODO: Add support for clauses which are valid for DISTRIBUTE
      // constructs. Static schedule is the default.
      auto schedule = omp::ClauseScheduleKind::Static;
      bool isOrdered = false;
      std::optional<omp::ScheduleModifier> scheduleMod;
      bool isSimd = false;
      llvm::omp::WorksharingLoopType workshareLoopType =
          llvm::omp::WorksharingLoopType::DistributeStaticLoop;
      bool loopNeedsBarrier = false;
      llvm::Value *chunk = nullptr;

      llvm::CanonicalLoopInfo *loopInfo =
          findCurrentLoopInfo(moduleTranslation);
      llvm::OpenMPIRBuilder::InsertPointOrErrorTy wsloopIP =
          ompBuilder->applyWorkshareLoop(
              ompLoc.DL, loopInfo, allocaIP, loopNeedsBarrier,
              convertToScheduleKind(schedule), chunk, isSimd,
              scheduleMod == omp::ScheduleModifier::monotonic,
              scheduleMod == omp::ScheduleModifier::nonmonotonic, isOrdered,
              workshareLoopType);

      if (!wsloopIP)
        return wsloopIP.takeError();
    }

    if (failed(cleanupPrivateVars(builder, moduleTranslation,
                                  distributeOp.getLoc(), privVarsInfo.llvmVars,
                                  privVarsInfo.privatizers)))
      return llvm::make_error<PreviouslyReportedError>();

    return llvm::Error::success();
  };

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      ompBuilder->createDistribute(ompLoc, allocaIP, bodyGenCB);

  if (failed(handleError(afterIP, opInst)))
    return failure();

  builder.restoreIP(*afterIP);

  if (doDistributeReduction) {
    // Process the reductions if required.
    return createReductionsAndCleanup(
        teamsOp, builder, moduleTranslation, allocaIP, reductionDecls,
        privateReductionVariables, isByRef,
        /*isNoWait*/ false, /*isTeamsReduction*/ true);
  }
  return success();
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

static void getTargetEntryUniqueInfo(llvm::TargetRegionEntryInfo &targetInfo,
                                     omp::TargetOp targetOp,
                                     llvm::StringRef parentName = "") {
  auto fileLoc = targetOp.getLoc()->findInstanceOf<FileLineColLoc>();

  assert(fileLoc && "No file found from location");
  StringRef fileName = fileLoc.getFilename().getValue();

  llvm::sys::fs::UniqueID id;
  uint64_t line = fileLoc.getLine();
  if (auto ec = llvm::sys::fs::getUniqueID(fileName, id)) {
    size_t fileHash = llvm::hash_value(fileName.str());
    size_t deviceId = 0xdeadf17e;
    targetInfo =
        llvm::TargetRegionEntryInfo(parentName, deviceId, fileHash, line);
  } else {
    targetInfo = llvm::TargetRegionEntryInfo(parentName, id.getDevice(),
                                             id.getFile(), line);
  }
}

static void
handleDeclareTargetMapVar(MapInfoData &mapData,
                          LLVM::ModuleTranslation &moduleTranslation,
                          llvm::IRBuilderBase &builder, llvm::Function *func) {
  assert(moduleTranslation.getOpenMPBuilder()->Config.isTargetDevice() &&
         "function only supported for target device codegen");
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
            load->moveBefore(insn->getIterator());
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
  assert(ompBuilder.Config.isTargetDevice() &&
         "function only supported for target device codegen");
  builder.restoreIP(allocaIP);

  omp::VariableCaptureKind capture = omp::VariableCaptureKind::ByRef;
  LLVM::TypeToLLVMIRTranslator typeToLLVMIRTranslator(
      ompBuilder.M.getContext());
  unsigned alignmentValue = 0;
  // Find the associated MapInfoData entry for the current input
  for (size_t i = 0; i < mapData.MapClause.size(); ++i)
    if (mapData.OriginalValue[i] == input) {
      auto mapOp = cast<omp::MapInfoOp>(mapData.MapClause[i]);
      capture = mapOp.getMapCaptureType();
      // Get information of alignment of mapped object
      alignmentValue = typeToLLVMIRTranslator.getPreferredAlignment(
          mapOp.getVarType(), ompBuilder.M.getDataLayout());
      break;
    }

  unsigned int allocaAS = ompBuilder.M.getDataLayout().getAllocaAddrSpace();
  unsigned int defaultAS =
      ompBuilder.M.getDataLayout().getProgramAddressSpace();

  // Create the alloca for the argument the current point.
  llvm::Value *v = builder.CreateAlloca(arg.getType(), allocaAS);

  if (allocaAS != defaultAS && arg.getType()->isPointerTy())
    v = builder.CreateAddrSpaceCast(v, builder.getPtrTy(defaultAS));

  builder.CreateStore(&arg, v);

  builder.restoreIP(codeGenIP);

  switch (capture) {
  case omp::VariableCaptureKind::ByCopy: {
    retVal = v;
    break;
  }
  case omp::VariableCaptureKind::ByRef: {
    llvm::LoadInst *loadInst = builder.CreateAlignedLoad(
        v->getType(), v,
        ompBuilder.M.getDataLayout().getPrefTypeAlign(v->getType()));
    // CreateAlignedLoad function creates similar LLVM IR:
    // %res = load ptr, ptr %input, align 8
    // This LLVM IR does not contain information about alignment
    // of the loaded value. We need to add !align metadata to unblock
    // optimizer. The existence of the !align metadata on the instruction
    // tells the optimizer that the value loaded is known to be aligned to
    // a boundary specified by the integer value in the metadata node.
    // Example:
    // %res = load ptr, ptr %input, align 8, !align !align_md_node
    //                                 ^         ^
    //                                 |         |
    //            alignment of %input address    |
    //                                           |
    //                                     alignment of %res object
    if (v->getType()->isPointerTy() && alignmentValue) {
      llvm::MDBuilder MDB(builder.getContext());
      loadInst->setMetadata(
          llvm::LLVMContext::MD_align,
          llvm::MDNode::get(builder.getContext(),
                            MDB.createConstant(llvm::ConstantInt::get(
                                llvm::Type::getInt64Ty(builder.getContext()),
                                alignmentValue))));
    }
    retVal = loadInst;

    break;
  }
  case omp::VariableCaptureKind::This:
  case omp::VariableCaptureKind::VLAType:
    // TODO: Consider returning error to use standard reporting for
    // unimplemented features.
    assert(false && "Currently unsupported capture kind");
    break;
  }

  return builder.saveIP();
}

/// Follow uses of `host_eval`-defined block arguments of the given `omp.target`
/// operation and populate output variables with their corresponding host value
/// (i.e. operand evaluated outside of the target region), based on their uses
/// inside of the target region.
///
/// Loop bounds and steps are only optionally populated, if output vectors are
/// provided.
static void
extractHostEvalClauses(omp::TargetOp targetOp, Value &numThreads,
                       Value &numTeamsLower, Value &numTeamsUpper,
                       Value &threadLimit,
                       llvm::SmallVectorImpl<Value> *lowerBounds = nullptr,
                       llvm::SmallVectorImpl<Value> *upperBounds = nullptr,
                       llvm::SmallVectorImpl<Value> *steps = nullptr) {
  auto blockArgIface = llvm::cast<omp::BlockArgOpenMPOpInterface>(*targetOp);
  for (auto item : llvm::zip_equal(targetOp.getHostEvalVars(),
                                   blockArgIface.getHostEvalBlockArgs())) {
    Value hostEvalVar = std::get<0>(item), blockArg = std::get<1>(item);

    for (Operation *user : blockArg.getUsers()) {
      llvm::TypeSwitch<Operation *>(user)
          .Case([&](omp::TeamsOp teamsOp) {
            if (teamsOp.getNumTeamsLower() == blockArg)
              numTeamsLower = hostEvalVar;
            else if (teamsOp.getNumTeamsUpper() == blockArg)
              numTeamsUpper = hostEvalVar;
            else if (teamsOp.getThreadLimit() == blockArg)
              threadLimit = hostEvalVar;
            else
              llvm_unreachable("unsupported host_eval use");
          })
          .Case([&](omp::ParallelOp parallelOp) {
            if (parallelOp.getNumThreads() == blockArg)
              numThreads = hostEvalVar;
            else
              llvm_unreachable("unsupported host_eval use");
          })
          .Case([&](omp::LoopNestOp loopOp) {
            auto processBounds =
                [&](OperandRange opBounds,
                    llvm::SmallVectorImpl<Value> *outBounds) -> bool {
              bool found = false;
              for (auto [i, lb] : llvm::enumerate(opBounds)) {
                if (lb == blockArg) {
                  found = true;
                  if (outBounds)
                    (*outBounds)[i] = hostEvalVar;
                }
              }
              return found;
            };
            bool found =
                processBounds(loopOp.getLoopLowerBounds(), lowerBounds);
            found = processBounds(loopOp.getLoopUpperBounds(), upperBounds) ||
                    found;
            found = processBounds(loopOp.getLoopSteps(), steps) || found;
            (void)found;
            assert(found && "unsupported host_eval use");
          })
          .Default([](Operation *) {
            llvm_unreachable("unsupported host_eval use");
          });
    }
  }
}

/// If \p op is of the given type parameter, return it casted to that type.
/// Otherwise, if its immediate parent operation (or some other higher-level
/// parent, if \p immediateParent is false) is of that type, return that parent
/// casted to the given type.
///
/// If \p op is \c null or neither it or its parent(s) are of the specified
/// type, return a \c null operation.
template <typename OpTy>
static OpTy castOrGetParentOfType(Operation *op, bool immediateParent = false) {
  if (!op)
    return OpTy();

  if (OpTy casted = dyn_cast<OpTy>(op))
    return casted;

  if (immediateParent)
    return dyn_cast_if_present<OpTy>(op->getParentOp());

  return op->getParentOfType<OpTy>();
}

/// If the given \p value is defined by an \c llvm.mlir.constant operation and
/// it is of an integer type, return its value.
static std::optional<int64_t> extractConstInteger(Value value) {
  if (!value)
    return std::nullopt;

  if (auto constOp =
          dyn_cast_if_present<LLVM::ConstantOp>(value.getDefiningOp()))
    if (auto constAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return constAttr.getInt();

  return std::nullopt;
}

static uint64_t getTypeByteSize(mlir::Type type, const DataLayout &dl) {
  uint64_t sizeInBits = dl.getTypeSizeInBits(type);
  uint64_t sizeInBytes = sizeInBits / 8;
  return sizeInBytes;
}

template <typename OpTy>
static uint64_t getReductionDataSize(OpTy &op) {
  if (op.getNumReductionVars() > 0) {
    SmallVector<omp::DeclareReductionOp> reductions;
    collectReductionDecls(op, reductions);

    llvm::SmallVector<mlir::Type> members;
    members.reserve(reductions.size());
    for (omp::DeclareReductionOp &red : reductions)
      members.push_back(red.getType());
    Operation *opp = op.getOperation();
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        opp->getContext(), members, /*isPacked=*/false);
    DataLayout dl = DataLayout(opp->getParentOfType<ModuleOp>());
    return getTypeByteSize(structType, dl);
  }
  return 0;
}

/// Populate default `MinTeams`, `MaxTeams` and `MaxThreads` to their default
/// values as stated by the corresponding clauses, if constant.
///
/// These default values must be set before the creation of the outlined LLVM
/// function for the target region, so that they can be used to initialize the
/// corresponding global `ConfigurationEnvironmentTy` structure.
static void
initTargetDefaultAttrs(omp::TargetOp targetOp, Operation *capturedOp,
                       llvm::OpenMPIRBuilder::TargetKernelDefaultAttrs &attrs,
                       bool isTargetDevice, bool isGPU) {
  // TODO: Handle constant 'if' clauses.

  Value numThreads, numTeamsLower, numTeamsUpper, threadLimit;
  if (!isTargetDevice) {
    extractHostEvalClauses(targetOp, numThreads, numTeamsLower, numTeamsUpper,
                           threadLimit);
  } else {
    // In the target device, values for these clauses are not passed as
    // host_eval, but instead evaluated prior to entry to the region. This
    // ensures values are mapped and available inside of the target region.
    if (auto teamsOp = castOrGetParentOfType<omp::TeamsOp>(capturedOp)) {
      numTeamsLower = teamsOp.getNumTeamsLower();
      numTeamsUpper = teamsOp.getNumTeamsUpper();
      threadLimit = teamsOp.getThreadLimit();
    }

    if (auto parallelOp = castOrGetParentOfType<omp::ParallelOp>(capturedOp))
      numThreads = parallelOp.getNumThreads();
  }

  // Handle clauses impacting the number of teams.

  int32_t minTeamsVal = 1, maxTeamsVal = -1;
  if (castOrGetParentOfType<omp::TeamsOp>(capturedOp)) {
    // TODO: Use `hostNumTeamsLower` to initialize `minTeamsVal`. For now, match
    // clang and set min and max to the same value.
    if (numTeamsUpper) {
      if (auto val = extractConstInteger(numTeamsUpper))
        minTeamsVal = maxTeamsVal = *val;
    } else {
      minTeamsVal = maxTeamsVal = 0;
    }
  } else if (castOrGetParentOfType<omp::ParallelOp>(capturedOp,
                                                    /*immediateParent=*/true) ||
             castOrGetParentOfType<omp::SimdOp>(capturedOp,
                                                /*immediateParent=*/true)) {
    minTeamsVal = maxTeamsVal = 1;
  } else {
    minTeamsVal = maxTeamsVal = -1;
  }

  // Handle clauses impacting the number of threads.

  auto setMaxValueFromClause = [](Value clauseValue, int32_t &result) {
    if (!clauseValue)
      return;

    if (auto val = extractConstInteger(clauseValue))
      result = *val;

    // Found an applicable clause, so it's not undefined. Mark as unknown
    // because it's not constant.
    if (result < 0)
      result = 0;
  };

  // Extract 'thread_limit' clause from 'target' and 'teams' directives.
  int32_t targetThreadLimitVal = -1, teamsThreadLimitVal = -1;
  setMaxValueFromClause(targetOp.getThreadLimit(), targetThreadLimitVal);
  setMaxValueFromClause(threadLimit, teamsThreadLimitVal);

  // Extract 'max_threads' clause from 'parallel' or set to 1 if it's SIMD.
  int32_t maxThreadsVal = -1;
  if (castOrGetParentOfType<omp::ParallelOp>(capturedOp))
    setMaxValueFromClause(numThreads, maxThreadsVal);
  else if (castOrGetParentOfType<omp::SimdOp>(capturedOp,
                                              /*immediateParent=*/true))
    maxThreadsVal = 1;

  // For max values, < 0 means unset, == 0 means set but unknown. Select the
  // minimum value between 'max_threads' and 'thread_limit' clauses that were
  // set.
  int32_t combinedMaxThreadsVal = targetThreadLimitVal;
  if (combinedMaxThreadsVal < 0 ||
      (teamsThreadLimitVal >= 0 && teamsThreadLimitVal < combinedMaxThreadsVal))
    combinedMaxThreadsVal = teamsThreadLimitVal;

  if (combinedMaxThreadsVal < 0 ||
      (maxThreadsVal >= 0 && maxThreadsVal < combinedMaxThreadsVal))
    combinedMaxThreadsVal = maxThreadsVal;

  int32_t reductionDataSize = 0;
  if (isGPU && capturedOp) {
    if (auto teamsOp = castOrGetParentOfType<omp::TeamsOp>(capturedOp))
      reductionDataSize = getReductionDataSize(teamsOp);
  }

  // Update kernel bounds structure for the `OpenMPIRBuilder` to use.
  omp::TargetRegionFlags kernelFlags = targetOp.getKernelExecFlags(capturedOp);
  assert(
      omp::bitEnumContainsAny(kernelFlags, omp::TargetRegionFlags::generic |
                                               omp::TargetRegionFlags::spmd) &&
      "invalid kernel flags");
  attrs.ExecFlags =
      omp::bitEnumContainsAny(kernelFlags, omp::TargetRegionFlags::generic)
          ? omp::bitEnumContainsAny(kernelFlags, omp::TargetRegionFlags::spmd)
                ? llvm::omp::OMP_TGT_EXEC_MODE_GENERIC_SPMD
                : llvm::omp::OMP_TGT_EXEC_MODE_GENERIC
          : llvm::omp::OMP_TGT_EXEC_MODE_SPMD;
  attrs.MinTeams = minTeamsVal;
  attrs.MaxTeams.front() = maxTeamsVal;
  attrs.MinThreads = 1;
  attrs.MaxThreads.front() = combinedMaxThreadsVal;
  attrs.ReductionDataSize = reductionDataSize;
  // TODO: Allow modified buffer length similar to
  // fopenmp-cuda-teams-reduction-recs-num flag in clang.
  if (attrs.ReductionDataSize != 0)
    attrs.ReductionBufferLength = 1024;
}

/// Gather LLVM runtime values for all clauses evaluated in the host that are
/// passed to the kernel invocation.
///
/// This function must be called only when compiling for the host. Also, it will
/// only provide correct results if it's called after the body of \c targetOp
/// has been fully generated.
static void
initTargetRuntimeAttrs(llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation,
                       omp::TargetOp targetOp, Operation *capturedOp,
                       llvm::OpenMPIRBuilder::TargetKernelRuntimeAttrs &attrs) {
  omp::LoopNestOp loopOp = castOrGetParentOfType<omp::LoopNestOp>(capturedOp);
  unsigned numLoops = loopOp ? loopOp.getNumLoops() : 0;

  Value numThreads, numTeamsLower, numTeamsUpper, teamsThreadLimit;
  llvm::SmallVector<Value> lowerBounds(numLoops), upperBounds(numLoops),
      steps(numLoops);
  extractHostEvalClauses(targetOp, numThreads, numTeamsLower, numTeamsUpper,
                         teamsThreadLimit, &lowerBounds, &upperBounds, &steps);

  // TODO: Handle constant 'if' clauses.
  if (Value targetThreadLimit = targetOp.getThreadLimit())
    attrs.TargetThreadLimit.front() =
        moduleTranslation.lookupValue(targetThreadLimit);

  if (numTeamsLower)
    attrs.MinTeams = moduleTranslation.lookupValue(numTeamsLower);

  if (numTeamsUpper)
    attrs.MaxTeams.front() = moduleTranslation.lookupValue(numTeamsUpper);

  if (teamsThreadLimit)
    attrs.TeamsThreadLimit.front() =
        moduleTranslation.lookupValue(teamsThreadLimit);

  if (numThreads)
    attrs.MaxThreads = moduleTranslation.lookupValue(numThreads);

  if (omp::bitEnumContainsAny(targetOp.getKernelExecFlags(capturedOp),
                              omp::TargetRegionFlags::trip_count)) {
    llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
    attrs.LoopTripCount = nullptr;

    // To calculate the trip count, we multiply together the trip counts of
    // every collapsed canonical loop. We don't need to create the loop nests
    // here, since we're only interested in the trip count.
    for (auto [loopLower, loopUpper, loopStep] :
         llvm::zip_equal(lowerBounds, upperBounds, steps)) {
      llvm::Value *lowerBound = moduleTranslation.lookupValue(loopLower);
      llvm::Value *upperBound = moduleTranslation.lookupValue(loopUpper);
      llvm::Value *step = moduleTranslation.lookupValue(loopStep);

      llvm::OpenMPIRBuilder::LocationDescription loc(builder);
      llvm::Value *tripCount = ompBuilder->calculateCanonicalLoopTripCount(
          loc, lowerBound, upperBound, step, /*IsSigned=*/true,
          loopOp.getLoopInclusive());

      if (!attrs.LoopTripCount) {
        attrs.LoopTripCount = tripCount;
        continue;
      }

      // TODO: Enable UndefinedSanitizer to diagnose an overflow here.
      attrs.LoopTripCount = builder.CreateMul(attrs.LoopTripCount, tripCount,
                                              {}, /*HasNUW=*/true);
    }
  }
}

static LogicalResult
convertOmpTarget(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  auto targetOp = cast<omp::TargetOp>(opInst);
  if (failed(checkImplementationStatus(opInst)))
    return failure();

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  bool isTargetDevice = ompBuilder->Config.isTargetDevice();
  bool isGPU = ompBuilder->Config.isGPU();

  auto parentFn = opInst.getParentOfType<LLVM::LLVMFuncOp>();
  auto argIface = cast<omp::BlockArgOpenMPOpInterface>(opInst);
  auto &targetRegion = targetOp.getRegion();
  // Holds the private vars that have been mapped along with the block argument
  // that corresponds to the MapInfoOp corresponding to the private var in
  // question. So, for instance:
  //
  // %10 = omp.map.info var_ptr(%6#0 : !fir.ref<!fir.box<!fir.heap<i32>>>, ..)
  // omp.target map_entries(%10 -> %arg0) private(@box.privatizer %6#0-> %arg1)
  //
  // Then, %10 has been created so that the descriptor can be used by the
  // privatizer @box.privatizer on the device side. Here we'd record {%6#0,
  // %arg0} in the mappedPrivateVars map.
  llvm::DenseMap<Value, Value> mappedPrivateVars;
  DataLayout dl = DataLayout(opInst.getParentOfType<ModuleOp>());
  SmallVector<Value> mapVars = targetOp.getMapVars();
  SmallVector<Value> hdaVars = targetOp.getHasDeviceAddrVars();
  ArrayRef<BlockArgument> mapBlockArgs = argIface.getMapBlockArgs();
  ArrayRef<BlockArgument> hdaBlockArgs = argIface.getHasDeviceAddrBlockArgs();
  llvm::Function *llvmOutlinedFn = nullptr;

  // TODO: It can also be false if a compile-time constant `false` IF clause is
  // specified.
  bool isOffloadEntry =
      isTargetDevice || !ompBuilder->Config.TargetTriples.empty();

  // For some private variables, the MapsForPrivatizedVariablesPass
  // creates MapInfoOp instances. Go through the private variables and
  // the mapped variables so that during codegeneration we are able
  // to quickly look up the corresponding map variable, if any for each
  // private variable.
  if (!targetOp.getPrivateVars().empty() && !targetOp.getMapVars().empty()) {
    OperandRange privateVars = targetOp.getPrivateVars();
    std::optional<ArrayAttr> privateSyms = targetOp.getPrivateSyms();
    std::optional<DenseI64ArrayAttr> privateMapIndices =
        targetOp.getPrivateMapsAttr();

    for (auto [privVarIdx, privVarSymPair] :
         llvm::enumerate(llvm::zip_equal(privateVars, *privateSyms))) {
      auto privVar = std::get<0>(privVarSymPair);
      auto privSym = std::get<1>(privVarSymPair);

      SymbolRefAttr privatizerName = llvm::cast<SymbolRefAttr>(privSym);
      omp::PrivateClauseOp privatizer =
          findPrivatizer(targetOp, privatizerName);

      if (!privatizer.needsMap())
        continue;

      mlir::Value mappedValue =
          targetOp.getMappedValueForPrivateVar(privVarIdx);
      assert(mappedValue && "Expected to find mapped value for a privatized "
                            "variable that needs mapping");

      // The MapInfoOp defining the map var isn't really needed later.
      // So, we don't store it in any datastructure. Instead, we just
      // do some sanity checks on it right now.
      auto mapInfoOp = mappedValue.getDefiningOp<omp::MapInfoOp>();
      [[maybe_unused]] Type varType = mapInfoOp.getVarType();

      // Check #1: Check that the type of the private variable matches
      // the type of the variable being mapped.
      if (!isa<LLVM::LLVMPointerType>(privVar.getType()))
        assert(
            varType == privVar.getType() &&
            "Type of private var doesn't match the type of the mapped value");

      // Ok, only 1 sanity check for now.
      // Record the block argument corresponding to this mapvar.
      mappedPrivateVars.insert(
          {privVar,
           targetRegion.getArgument(argIface.getMapBlockArgsStart() +
                                    (*privateMapIndices)[privVarIdx])});
    }
  }

  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP)
      -> llvm::OpenMPIRBuilder::InsertPointOrErrorTy {
    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    builder.SetCurrentDebugLocation(llvm::DebugLoc());
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

    for (auto [arg, mapOp] : llvm::zip_equal(mapBlockArgs, mapVars)) {
      auto mapInfoOp = cast<omp::MapInfoOp>(mapOp.getDefiningOp());
      llvm::Value *mapOpValue =
          moduleTranslation.lookupValue(mapInfoOp.getVarPtr());
      moduleTranslation.mapValue(arg, mapOpValue);
    }
    for (auto [arg, mapOp] : llvm::zip_equal(hdaBlockArgs, hdaVars)) {
      auto mapInfoOp = cast<omp::MapInfoOp>(mapOp.getDefiningOp());
      llvm::Value *mapOpValue =
          moduleTranslation.lookupValue(mapInfoOp.getVarPtr());
      moduleTranslation.mapValue(arg, mapOpValue);
    }

    // Do privatization after moduleTranslation has already recorded
    // mapped values.
    PrivateVarsInfo privateVarsInfo(targetOp);

    llvm::Expected<llvm::BasicBlock *> afterAllocas =
        allocatePrivateVars(builder, moduleTranslation, privateVarsInfo,
                            allocaIP, &mappedPrivateVars);

    if (failed(handleError(afterAllocas, *targetOp)))
      return llvm::make_error<PreviouslyReportedError>();

    builder.restoreIP(codeGenIP);
    if (handleError(initPrivateVars(builder, moduleTranslation, privateVarsInfo,
                                    &mappedPrivateVars),
                    *targetOp)
            .failed())
      return llvm::make_error<PreviouslyReportedError>();

    if (failed(copyFirstPrivateVars(
            targetOp, builder, moduleTranslation, privateVarsInfo.mlirVars,
            privateVarsInfo.llvmVars, privateVarsInfo.privatizers,
            targetOp.getPrivateNeedsBarrier(), &mappedPrivateVars)))
      return llvm::make_error<PreviouslyReportedError>();

    SmallVector<Region *> privateCleanupRegions;
    llvm::transform(privateVarsInfo.privatizers,
                    std::back_inserter(privateCleanupRegions),
                    [](omp::PrivateClauseOp privatizer) {
                      return &privatizer.getDeallocRegion();
                    });

    llvm::Expected<llvm::BasicBlock *> exitBlock = convertOmpOpRegions(
        targetRegion, "omp.target", builder, moduleTranslation);

    if (!exitBlock)
      return exitBlock.takeError();

    builder.SetInsertPoint(*exitBlock);
    if (!privateCleanupRegions.empty()) {
      if (failed(inlineOmpRegionCleanup(
              privateCleanupRegions, privateVarsInfo.llvmVars,
              moduleTranslation, builder, "omp.targetop.private.cleanup",
              /*shouldLoadCleanupRegionArg=*/false))) {
        return llvm::createStringError(
            "failed to inline `dealloc` region of `omp.private` "
            "op in the target region");
      }
      return builder.saveIP();
    }

    return InsertPointTy(exitBlock.get(), exitBlock.get()->end());
  };

  StringRef parentName = parentFn.getName();

  llvm::TargetRegionEntryInfo entryInfo;

  getTargetEntryUniqueInfo(entryInfo, targetOp, parentName);

  MapInfoData mapData;
  collectMapDataFromMapOperands(mapData, mapVars, moduleTranslation, dl,
                                builder, /*useDevPtrOperands=*/{},
                                /*useDevAddrOperands=*/{}, hdaVars);

  MapInfosTy combinedInfos;
  auto genMapInfoCB =
      [&](llvm::OpenMPIRBuilder::InsertPointTy codeGenIP) -> MapInfosTy & {
    builder.restoreIP(codeGenIP);
    genMapInfos(builder, moduleTranslation, dl, combinedInfos, mapData, true);
    return combinedInfos;
  };

  auto argAccessorCB = [&](llvm::Argument &arg, llvm::Value *input,
                           llvm::Value *&retVal, InsertPointTy allocaIP,
                           InsertPointTy codeGenIP)
      -> llvm::OpenMPIRBuilder::InsertPointOrErrorTy {
    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    builder.SetCurrentDebugLocation(llvm::DebugLoc());
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

  llvm::OpenMPIRBuilder::TargetKernelRuntimeAttrs runtimeAttrs;
  llvm::OpenMPIRBuilder::TargetKernelDefaultAttrs defaultAttrs;
  Operation *targetCapturedOp = targetOp.getInnermostCapturedOmpOp();
  initTargetDefaultAttrs(targetOp, targetCapturedOp, defaultAttrs,
                         isTargetDevice, isGPU);

  // Collect host-evaluated values needed to properly launch the kernel from the
  // host.
  if (!isTargetDevice)
    initTargetRuntimeAttrs(builder, moduleTranslation, targetOp,
                           targetCapturedOp, runtimeAttrs);

  // Pass host-evaluated values as parameters to the kernel / host fallback,
  // except if they are constants. In any case, map the MLIR block argument to
  // the corresponding LLVM values.
  llvm::SmallVector<llvm::Value *, 4> kernelInput;
  SmallVector<Value> hostEvalVars = targetOp.getHostEvalVars();
  ArrayRef<BlockArgument> hostEvalBlockArgs = argIface.getHostEvalBlockArgs();
  for (auto [arg, var] : llvm::zip_equal(hostEvalBlockArgs, hostEvalVars)) {
    llvm::Value *value = moduleTranslation.lookupValue(var);
    moduleTranslation.mapValue(arg, value);

    if (!llvm::isa<llvm::Constant>(value))
      kernelInput.push_back(value);
  }

  for (size_t i = 0, e = mapData.OriginalValue.size(); i != e; ++i) {
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

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  llvm::OpenMPIRBuilder::TargetDataInfo info(
      /*RequiresDevicePointerInfo=*/false,
      /*SeparateBeginEndCalls=*/true);

  auto customMapperCB =
      [&](unsigned int i) -> llvm::Expected<llvm::Function *> {
    if (!combinedInfos.Mappers[i])
      return nullptr;
    info.HasMapper = true;
    return getOrCreateUserDefinedMapperFunc(combinedInfos.Mappers[i], builder,
                                            moduleTranslation);
  };

  llvm::Value *ifCond = nullptr;
  if (Value targetIfCond = targetOp.getIfExpr())
    ifCond = moduleTranslation.lookupValue(targetIfCond);

  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createTarget(
          ompLoc, isOffloadEntry, allocaIP, builder.saveIP(), info, entryInfo,
          defaultAttrs, runtimeAttrs, ifCond, kernelInput, genMapInfoCB, bodyCB,
          argAccessorCB, customMapperCB, dds, targetOp.getNowait());

  if (failed(handleError(afterIP, opInst)))
    return failure();

  builder.restoreIP(*afterIP);

  // Remap access operations to declare target reference pointers for the
  // device, essentially generating extra loadop's as necessary
  if (moduleTranslation.getOpenMPBuilder()->Config.isTargetDevice())
    handleDeclareTargetMapVar(mapData, moduleTranslation, builder,
                              llvmOutlinedFn);

  return success();
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

  // Certain operations return results, and whether utilised in host or
  // target there is a chance an LLVM Dialect operation depends on it
  // by taking it in as an operand, so we must always lower these in
  // some manner or result in an ICE (whether they end up in a no-op
  // or otherwise).
  if (mlir::isa<omp::ThreadprivateOp>(op))
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

/// Given an OpenMP MLIR operation, create the corresponding LLVM IR (including
/// OpenMP runtime calls).
static LogicalResult
convertHostOrTargetOperation(Operation *op, llvm::IRBuilderBase &builder,
                             LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // For each loop, introduce one stack frame to hold loop information. Ensure
  // this is only done for the outermost loop wrapper to prevent introducing
  // multiple stack frames for a single loop. Initially set to null, the loop
  // information structure is initialized during translation of the nested
  // omp.loop_nest operation, making it available to translation of all loop
  // wrappers after their body has been successfully translated.
  bool isOutermostLoopWrapper =
      isa_and_present<omp::LoopWrapperInterface>(op) &&
      !dyn_cast_if_present<omp::LoopWrapperInterface>(op->getParentOp());

  if (isOutermostLoopWrapper)
    moduleTranslation.stackPush<OpenMPLoopInfoStackFrame>();

  auto result =
      llvm::TypeSwitch<Operation *, LogicalResult>(op)
          .Case([&](omp::BarrierOp op) -> LogicalResult {
            if (failed(checkImplementationStatus(*op)))
              return failure();

            llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
                ompBuilder->createBarrier(builder.saveIP(),
                                          llvm::omp::OMPD_barrier);
            return handleError(afterIP, *op);
          })
          .Case([&](omp::TaskyieldOp op) {
            if (failed(checkImplementationStatus(*op)))
              return failure();

            ompBuilder->createTaskyield(builder.saveIP());
            return success();
          })
          .Case([&](omp::FlushOp op) {
            if (failed(checkImplementationStatus(*op)))
              return failure();

            // No support in Openmp runtime function (__kmpc_flush) to accept
            // the argument list.
            // OpenMP standard states the following:
            //  "An implementation may implement a flush with a list by ignoring
            //   the list, and treating it the same as a flush without a list."
            //
            // The argument list is discarded so that, flush with a list is
            // treated same as a flush without a list.
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
          .Case([&](omp::CancelOp op) {
            return convertOmpCancel(op, builder, moduleTranslation);
          })
          .Case([&](omp::CancellationPointOp op) {
            return convertOmpCancellationPoint(op, builder, moduleTranslation);
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
          .Case<omp::YieldOp, omp::TerminatorOp, omp::DeclareMapperOp,
                omp::DeclareMapperInfoOp, omp::DeclareReductionOp,
                omp::CriticalDeclareOp>([](auto op) {
            // `yield` and `terminator` can be just omitted. The block structure
            // was created in the region that handles their parent operation.
            // `declare_reduction` will be used by reductions and is not
            // converted directly, skip it.
            // `declare_mapper` and `declare_mapper.info` are handled whenever
            // they are referred to through a `map` clause.
            // `critical.declare` is only used to declare names of critical
            // sections which will be used by `critical` ops and hence can be
            // ignored for lowering. The OpenMP IRBuilder will create unique
            // name for critical section names.
            return success();
          })
          .Case([&](omp::ThreadprivateOp) {
            return convertOmpThreadprivate(*op, builder, moduleTranslation);
          })
          .Case<omp::TargetDataOp, omp::TargetEnterDataOp,
                omp::TargetExitDataOp, omp::TargetUpdateOp>([&](auto op) {
            return convertOmpTargetData(op, builder, moduleTranslation);
          })
          .Case([&](omp::TargetOp) {
            return convertOmpTarget(*op, builder, moduleTranslation);
          })
          .Case([&](omp::DistributeOp) {
            return convertOmpDistribute(*op, builder, moduleTranslation);
          })
          .Case([&](omp::LoopNestOp) {
            return convertOmpLoopNest(*op, builder, moduleTranslation);
          })
          .Case<omp::MapInfoOp, omp::MapBoundsOp, omp::PrivateClauseOp>(
              [&](auto op) {
                // No-op, should be handled by relevant owning operations e.g.
                // TargetOp, TargetEnterDataOp, TargetExitDataOp, TargetDataOp
                // etc. and then discarded
                return success();
              })
          .Default([&](Operation *inst) {
            return inst->emitError()
                   << "not yet implemented: " << inst->getName();
          });

  if (isOutermostLoopWrapper)
    moduleTranslation.stackPop();

  return result;
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

          // Non-target ops might nest target-related ops, therefore, we
          // translate them as non-OpenMP scopes. Translating them is needed by
          // nested target-related ops since they might need LLVM values defined
          // in their parent non-target ops.
          if (isa<omp::OpenMPDialect>(oper->getDialect()) &&
              oper->getParentOfType<LLVM::LLVMFuncOp>() &&
              !oper->getRegions().empty()) {
            if (auto blockArgsIface =
                    dyn_cast<omp::BlockArgOpenMPOpInterface>(oper))
              forwardArgs(moduleTranslation, blockArgsIface);
            else {
              // Here we map entry block arguments of
              // non-BlockArgOpenMPOpInterface ops if they can be encountered
              // inside of a function and they define any of these arguments.
              if (isa<mlir::omp::AtomicUpdateOp>(oper))
                for (auto [operand, arg] :
                     llvm::zip_equal(oper->getOperands(),
                                     oper->getRegion(0).getArguments())) {
                  moduleTranslation.mapValue(
                      arg, builder.CreateLoad(
                               moduleTranslation.convertType(arg.getType()),
                               moduleTranslation.lookupValue(operand)));
                }
            }

            if (auto loopNest = dyn_cast<omp::LoopNestOp>(oper)) {
              assert(builder.GetInsertBlock() &&
                     "No insert block is set for the builder");
              for (auto iv : loopNest.getIVs()) {
                // Map iv to an undefined value just to keep the IR validity.
                moduleTranslation.mapValue(
                    iv, llvm::PoisonValue::get(
                            moduleTranslation.convertType(iv.getType())));
              }
            }

            for (Region &region : oper->getRegions()) {
              // Regions are fake in the sense that they are not a truthful
              // translation of the OpenMP construct being converted (e.g. no
              // OpenMP runtime calls will be generated). We just need this to
              // prepare the kernel invocation args.
              SmallVector<llvm::PHINode *> phis;
              auto result = convertOmpOpRegions(
                  region, oper->getName().getStringRef().str() + ".fake.region",
                  builder, moduleTranslation, &phis);
              if (failed(handleError(result, *oper)))
                return WalkResult::interrupt();

              builder.SetInsertPoint(result.get(), result.get()->end());
            }

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
