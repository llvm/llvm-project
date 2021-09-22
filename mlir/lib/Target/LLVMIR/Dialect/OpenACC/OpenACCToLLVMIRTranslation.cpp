//===- OpenACCToLLVMIRTranslation.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR OpenACC dialect and LLVM
// IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/OpenACC/OpenACCToLLVMIRTranslation.h"
#include "mlir/Conversion/OpenACCToLLVM/ConvertOpenACCToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

using OpenACCIRBuilder = llvm::OpenMPIRBuilder;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Flag values are extracted from openmp/libomptarget/include/omptarget.h and
/// mapped to corresponding OpenACC flags.
static constexpr uint64_t kCreateFlag = 0x000;
static constexpr uint64_t kDeviceCopyinFlag = 0x001;
static constexpr uint64_t kHostCopyoutFlag = 0x002;
static constexpr uint64_t kCopyFlag = kDeviceCopyinFlag | kHostCopyoutFlag;
static constexpr uint64_t kPresentFlag = 0x1000;
static constexpr uint64_t kDeleteFlag = 0x008;
// Runtime extension to implement the OpenACC second reference counter.
static constexpr uint64_t kHoldFlag = 0x2000;

/// Default value for the device id
static constexpr int64_t kDefaultDevice = -1;

/// Create a constant string location from the MLIR Location information.
static llvm::Constant *createSourceLocStrFromLocation(Location loc,
                                                      OpenACCIRBuilder &builder,
                                                      StringRef name) {
  if (auto fileLoc = loc.dyn_cast<FileLineColLoc>()) {
    StringRef fileName = fileLoc.getFilename();
    unsigned lineNo = fileLoc.getLine();
    unsigned colNo = fileLoc.getColumn();
    return builder.getOrCreateSrcLocStr(name, fileName, lineNo, colNo);
  } else {
    std::string locStr;
    llvm::raw_string_ostream locOS(locStr);
    locOS << loc;
    return builder.getOrCreateSrcLocStr(locOS.str());
  }
}

/// Create the location struct from the operation location information.
static llvm::Value *createSourceLocationInfo(OpenACCIRBuilder &builder,
                                             Operation *op) {
  auto loc = op->getLoc();
  auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
  StringRef funcName = funcOp ? funcOp.getName() : "unknown";
  llvm::Constant *locStr =
      createSourceLocStrFromLocation(loc, builder, funcName);
  return builder.getOrCreateIdent(locStr);
}

/// Create a constant string representing the mapping information extracted from
/// the MLIR location information.
static llvm::Constant *createMappingInformation(Location loc,
                                                OpenACCIRBuilder &builder) {
  if (auto nameLoc = loc.dyn_cast<NameLoc>()) {
    StringRef name = nameLoc.getName();
    return createSourceLocStrFromLocation(nameLoc.getChildLoc(), builder, name);
  } else {
    return createSourceLocStrFromLocation(loc, builder, "unknown");
  }
}

/// Return the runtime function used to lower the given operation.
static llvm::Function *getAssociatedFunction(OpenACCIRBuilder &builder,
                                             Operation *op) {
  return llvm::TypeSwitch<Operation *, llvm::Function *>(op)
      .Case([&](acc::EnterDataOp) {
        return builder.getOrCreateRuntimeFunctionPtr(
            llvm::omp::OMPRTL___tgt_target_data_begin_mapper);
      })
      .Case([&](acc::ExitDataOp) {
        return builder.getOrCreateRuntimeFunctionPtr(
            llvm::omp::OMPRTL___tgt_target_data_end_mapper);
      })
      .Case([&](acc::UpdateOp) {
        return builder.getOrCreateRuntimeFunctionPtr(
            llvm::omp::OMPRTL___tgt_target_data_update_mapper);
      });
  llvm_unreachable("Unknown OpenACC operation");
}

/// Computes the size of type in bytes.
static llvm::Value *getSizeInBytes(llvm::IRBuilderBase &builder,
                                   llvm::Value *basePtr) {
  llvm::LLVMContext &ctx = builder.getContext();
  llvm::Value *null =
      llvm::Constant::getNullValue(basePtr->getType()->getPointerTo());
  llvm::Value *sizeGep =
      builder.CreateGEP(basePtr->getType(), null, builder.getInt32(1));
  llvm::Value *sizePtrToInt =
      builder.CreatePtrToInt(sizeGep, llvm::Type::getInt64Ty(ctx));
  return sizePtrToInt;
}

/// Extract pointer, size and mapping information from operands
/// to populate the future functions arguments.
static LogicalResult
processOperands(llvm::IRBuilderBase &builder,
                LLVM::ModuleTranslation &moduleTranslation, Operation *op,
                ValueRange operands, unsigned totalNbOperand,
                uint64_t operandFlag, SmallVector<uint64_t> &flags,
                SmallVectorImpl<llvm::Constant *> &names, unsigned &index,
                struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::LLVMContext &ctx = builder.getContext();
  auto *i8PtrTy = llvm::Type::getInt8PtrTy(ctx);
  auto *arrI8PtrTy = llvm::ArrayType::get(i8PtrTy, totalNbOperand);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *arrI64Ty = llvm::ArrayType::get(i64Ty, totalNbOperand);

  for (Value data : operands) {
    llvm::Value *dataValue = moduleTranslation.lookupValue(data);

    llvm::Value *dataPtrBase;
    llvm::Value *dataPtr;
    llvm::Value *dataSize;

    // Handle operands that were converted to DataDescriptor.
    if (DataDescriptor::isValid(data)) {
      dataPtrBase =
          builder.CreateExtractValue(dataValue, kPtrBasePosInDataDescriptor);
      dataPtr = builder.CreateExtractValue(dataValue, kPtrPosInDataDescriptor);
      dataSize =
          builder.CreateExtractValue(dataValue, kSizePosInDataDescriptor);
    } else if (data.getType().isa<LLVM::LLVMPointerType>()) {
      dataPtrBase = dataValue;
      dataPtr = dataValue;
      dataSize = getSizeInBytes(builder, dataValue);
    } else {
      return op->emitOpError()
             << "Data operand must be legalized before translation."
             << "Unsupported type: " << data.getType();
    }

    // Store base pointer extracted from operand into the i-th position of
    // argBase.
    llvm::Value *ptrBaseGEP = builder.CreateInBoundsGEP(
        arrI8PtrTy, mapperAllocas.ArgsBase,
        {builder.getInt32(0), builder.getInt32(index)});
    llvm::Value *ptrBaseCast = builder.CreateBitCast(
        ptrBaseGEP, dataPtrBase->getType()->getPointerTo());
    builder.CreateStore(dataPtrBase, ptrBaseCast);

    // Store pointer extracted from operand into the i-th position of args.
    llvm::Value *ptrGEP = builder.CreateInBoundsGEP(
        arrI8PtrTy, mapperAllocas.Args,
        {builder.getInt32(0), builder.getInt32(index)});
    llvm::Value *ptrCast =
        builder.CreateBitCast(ptrGEP, dataPtr->getType()->getPointerTo());
    builder.CreateStore(dataPtr, ptrCast);

    // Store size extracted from operand into the i-th position of argSizes.
    llvm::Value *sizeGEP = builder.CreateInBoundsGEP(
        arrI64Ty, mapperAllocas.ArgSizes,
        {builder.getInt32(0), builder.getInt32(index)});
    builder.CreateStore(dataSize, sizeGEP);

    flags.push_back(operandFlag);
    llvm::Constant *mapName =
        createMappingInformation(data.getLoc(), *accBuilder);
    names.push_back(mapName);
    ++index;
  }
  return success();
}

/// Process data operands from acc::EnterDataOp
static LogicalResult
processDataOperands(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    acc::EnterDataOp op, SmallVector<uint64_t> &flags,
                    SmallVectorImpl<llvm::Constant *> &names,
                    struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  // TODO add `create_zero` and `attach` operands

  unsigned index = 0;

  // Create operands are handled as `alloc` call.
  if (failed(processOperands(builder, moduleTranslation, op,
                             op.createOperands(), op.getNumDataOperands(),
                             kCreateFlag, flags, names, index, mapperAllocas)))
    return failure();

  // Copyin operands are handled as `to` call.
  if (failed(processOperands(builder, moduleTranslation, op,
                             op.copyinOperands(), op.getNumDataOperands(),
                             kDeviceCopyinFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  return success();
}

/// Process data operands from acc::ExitDataOp
static LogicalResult
processDataOperands(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    acc::ExitDataOp op, SmallVector<uint64_t> &flags,
                    SmallVectorImpl<llvm::Constant *> &names,
                    struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  // TODO add `detach` operands

  unsigned index = 0;

  // Delete operands are handled as `delete` call.
  if (failed(processOperands(builder, moduleTranslation, op,
                             op.deleteOperands(), op.getNumDataOperands(),
                             kDeleteFlag, flags, names, index, mapperAllocas)))
    return failure();

  // Copyout operands are handled as `from` call.
  if (failed(processOperands(builder, moduleTranslation, op,
                             op.copyoutOperands(), op.getNumDataOperands(),
                             kHostCopyoutFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  return success();
}

/// Process data operands from acc::UpdateOp
static LogicalResult
processDataOperands(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    acc::UpdateOp op, SmallVector<uint64_t> &flags,
                    SmallVectorImpl<llvm::Constant *> &names,
                    struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  unsigned index = 0;

  // Host operands are handled as `from` call.
  if (failed(processOperands(builder, moduleTranslation, op, op.hostOperands(),
                             op.getNumDataOperands(), kHostCopyoutFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Device operands are handled as `to` call.
  if (failed(processOperands(builder, moduleTranslation, op,
                             op.deviceOperands(), op.getNumDataOperands(),
                             kDeviceCopyinFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Conversion functions
//===----------------------------------------------------------------------===//

/// Converts an OpenACC data operation into LLVM IR.
static LogicalResult convertDataOp(acc::DataOp &op,
                                   llvm::IRBuilderBase &builder,
                                   LLVM::ModuleTranslation &moduleTranslation) {
  llvm::LLVMContext &ctx = builder.getContext();
  auto enclosingFuncOp = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());

  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();

  llvm::Value *srcLocInfo = createSourceLocationInfo(*accBuilder, op);

  llvm::Function *beginMapperFunc = accBuilder->getOrCreateRuntimeFunctionPtr(
      llvm::omp::OMPRTL___tgt_target_data_begin_mapper);

  llvm::Function *endMapperFunc = accBuilder->getOrCreateRuntimeFunctionPtr(
      llvm::omp::OMPRTL___tgt_target_data_end_mapper);

  // Number of arguments in the data operation.
  unsigned totalNbOperand = op.getNumDataOperands();

  struct OpenACCIRBuilder::MapperAllocas mapperAllocas;
  OpenACCIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  accBuilder->createMapperAllocas(builder.saveIP(), allocaIP, totalNbOperand,
                                  mapperAllocas);

  SmallVector<uint64_t> flags;
  SmallVector<llvm::Constant *> names;
  unsigned index = 0;

  // TODO handle no_create, deviceptr and attach operands.

  if (failed(processOperands(builder, moduleTranslation, op, op.copyOperands(),
                             totalNbOperand, kCopyFlag | kHoldFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  if (failed(processOperands(
          builder, moduleTranslation, op, op.copyinOperands(), totalNbOperand,
          kDeviceCopyinFlag | kHoldFlag, flags, names, index, mapperAllocas)))
    return failure();

  // TODO copyin readonly currenlty handled as copyin. Update when extension
  // available.
  if (failed(processOperands(builder, moduleTranslation, op,
                             op.copyinReadonlyOperands(), totalNbOperand,
                             kDeviceCopyinFlag | kHoldFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  if (failed(processOperands(
          builder, moduleTranslation, op, op.copyoutOperands(), totalNbOperand,
          kHostCopyoutFlag | kHoldFlag, flags, names, index, mapperAllocas)))
    return failure();

  // TODO copyout zero currenlty handled as copyout. Update when extension
  // available.
  if (failed(processOperands(builder, moduleTranslation, op,
                             op.copyoutZeroOperands(), totalNbOperand,
                             kHostCopyoutFlag | kHoldFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  if (failed(processOperands(builder, moduleTranslation, op,
                             op.createOperands(), totalNbOperand,
                             kCreateFlag | kHoldFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  // TODO create zero currenlty handled as create. Update when extension
  // available.
  if (failed(processOperands(builder, moduleTranslation, op,
                             op.createZeroOperands(), totalNbOperand,
                             kCreateFlag | kHoldFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  if (failed(processOperands(builder, moduleTranslation, op,
                             op.presentOperands(), totalNbOperand,
                             kPresentFlag | kHoldFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  llvm::GlobalVariable *maptypes =
      accBuilder->createOffloadMaptypes(flags, ".offload_maptypes");
  llvm::Value *maptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), totalNbOperand),
      maptypes, /*Idx0=*/0, /*Idx1=*/0);

  llvm::GlobalVariable *mapnames =
      accBuilder->createOffloadMapnames(names, ".offload_mapnames");
  llvm::Value *mapnamesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt8PtrTy(ctx), totalNbOperand),
      mapnames, /*Idx0=*/0, /*Idx1=*/0);

  // Create call to start the data region.
  accBuilder->emitMapperCall(builder.saveIP(), beginMapperFunc, srcLocInfo,
                             maptypesArg, mapnamesArg, mapperAllocas,
                             kDefaultDevice, totalNbOperand);

  // Convert the region.
  llvm::BasicBlock *entryBlock = nullptr;

  for (Block &bb : op.region()) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        ctx, "acc.data", builder.GetInsertBlock()->getParent());
    if (entryBlock == nullptr)
      entryBlock = llvmBB;
    moduleTranslation.mapBlock(&bb, llvmBB);
  }

  auto afterDataRegion = builder.saveIP();

  llvm::BranchInst *sourceTerminator = builder.CreateBr(entryBlock);

  builder.restoreIP(afterDataRegion);
  llvm::BasicBlock *endDataBlock = llvm::BasicBlock::Create(
      ctx, "acc.end_data", builder.GetInsertBlock()->getParent());

  SetVector<Block *> blocks =
      LLVM::detail::getTopologicallySortedBlocks(op.region());
  for (Block *bb : blocks) {
    llvm::BasicBlock *llvmBB = moduleTranslation.lookupBlock(bb);
    if (bb->isEntryBlock()) {
      assert(sourceTerminator->getNumSuccessors() == 1 &&
             "provided entry block has multiple successors");
      sourceTerminator->setSuccessor(0, llvmBB);
    }

    if (failed(
            moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder))) {
      return failure();
    }

    if (isa<acc::TerminatorOp, acc::YieldOp>(bb->getTerminator()))
      builder.CreateBr(endDataBlock);
  }

  // Create call to end the data region.
  builder.SetInsertPoint(endDataBlock);
  accBuilder->emitMapperCall(builder.saveIP(), endMapperFunc, srcLocInfo,
                             maptypesArg, mapnamesArg, mapperAllocas,
                             kDefaultDevice, totalNbOperand);

  return success();
}

/// Converts an OpenACC standalone data operation into LLVM IR.
template <typename OpTy>
static LogicalResult
convertStandaloneDataOp(OpTy &op, llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  auto enclosingFuncOp =
      op.getOperation()->template getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());

  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();

  auto *srcLocInfo = createSourceLocationInfo(*accBuilder, op);
  auto *mapperFunc = getAssociatedFunction(*accBuilder, op);

  // Number of arguments in the enter_data operation.
  unsigned totalNbOperand = op.getNumDataOperands();

  llvm::LLVMContext &ctx = builder.getContext();

  struct OpenACCIRBuilder::MapperAllocas mapperAllocas;
  OpenACCIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  accBuilder->createMapperAllocas(builder.saveIP(), allocaIP, totalNbOperand,
                                  mapperAllocas);

  SmallVector<uint64_t> flags;
  SmallVector<llvm::Constant *> names;

  if (failed(processDataOperands(builder, moduleTranslation, op, flags, names,
                                 mapperAllocas)))
    return failure();

  llvm::GlobalVariable *maptypes =
      accBuilder->createOffloadMaptypes(flags, ".offload_maptypes");
  llvm::Value *maptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), totalNbOperand),
      maptypes, /*Idx0=*/0, /*Idx1=*/0);

  llvm::GlobalVariable *mapnames =
      accBuilder->createOffloadMapnames(names, ".offload_mapnames");
  llvm::Value *mapnamesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt8PtrTy(ctx), totalNbOperand),
      mapnames, /*Idx0=*/0, /*Idx1=*/0);

  accBuilder->emitMapperCall(builder.saveIP(), mapperFunc, srcLocInfo,
                             maptypesArg, mapnamesArg, mapperAllocas,
                             kDefaultDevice, totalNbOperand);

  return success();
}

namespace {

/// Implementation of the dialect interface that converts operations belonging
/// to the OpenACC dialect to LLVM IR.
class OpenACCDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final;
};

} // end namespace

/// Given an OpenACC MLIR operation, create the corresponding LLVM IR
/// (including OpenACC runtime calls).
LogicalResult OpenACCDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](acc::DataOp dataOp) {
        return convertDataOp(dataOp, builder, moduleTranslation);
      })
      .Case([&](acc::EnterDataOp enterDataOp) {
        return convertStandaloneDataOp<acc::EnterDataOp>(enterDataOp, builder,
                                                         moduleTranslation);
      })
      .Case([&](acc::ExitDataOp exitDataOp) {
        return convertStandaloneDataOp<acc::ExitDataOp>(exitDataOp, builder,
                                                        moduleTranslation);
      })
      .Case([&](acc::UpdateOp updateOp) {
        return convertStandaloneDataOp<acc::UpdateOp>(updateOp, builder,
                                                      moduleTranslation);
      })
      .Case<acc::TerminatorOp, acc::YieldOp>([](auto op) {
        // `yield` and `terminator` can be just omitted. The block structure was
        // created in the function that handles their parent operation.
        assert(op->getNumOperands() == 0 &&
               "unexpected OpenACC terminator with operands");
        return success();
      })
      .Default([&](Operation *op) {
        return op->emitError("unsupported OpenACC operation: ")
               << op->getName();
      });
}

void mlir::registerOpenACCDialectTranslation(DialectRegistry &registry) {
  registry.insert<acc::OpenACCDialect>();
  registry.addDialectInterface<acc::OpenACCDialect,
                               OpenACCDialectLLVMIRTranslationInterface>();
}

void mlir::registerOpenACCDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerOpenACCDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
