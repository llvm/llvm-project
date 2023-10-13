//===- GENXToLLVMIRTranslation.cpp - Translate GENX to LLVM IR ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR GENX dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/GENX/GENXToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/Dialect/GENX/GenIntrinsics.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

// Create a call to SPIR device function.
static llvm::Value *createDeviceFunctionCall(llvm::IRBuilderBase &builder,
                                             StringRef fnName,
                                             llvm::Type *retType,
                                             ArrayRef<llvm::Type *> argTypes,
                                             ArrayRef<llvm::Value *> args) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  auto *functionType =
      llvm::FunctionType::get(retType, argTypes, /*isVarArg*/ false);
  auto *fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fnName, functionType).getCallee());
  fn->setCallingConv(llvm::CallingConv::SPIR_FUNC);
  return builder.CreateCall(fn, args);
}

// Create a call to SPIR a "atomic_work_item_fence" function.
static void createFence(llvm::IRBuilderBase &builder,
                        GENX::MemoryFenceFlag flags, GENX::MemoryOrder order,
                        GENX::MemoryScope scope) {
  std::string fnName =
      "_Z22atomic_work_item_fencej12memory_order12memory_scope";
  llvm::IntegerType *i32Ty = builder.getInt32Ty();
  createDeviceFunctionCall(
      builder, fnName, builder.getVoidTy(), {i32Ty, i32Ty, i32Ty},
      {llvm::ConstantInt::get(i32Ty, static_cast<int>(flags)),
       llvm::ConstantInt::get(i32Ty, static_cast<int>(order)),
       llvm::ConstantInt::get(i32Ty, static_cast<int>(scope))});
}

// Create a call to SPIR a "sub_group_shuffle" function.
static llvm::Value *createSubGroupShuffle(llvm::IRBuilderBase &builder,
                                          llvm::Value *value, llvm::Value *mask,
                                          GENX::ShflKind kind) {
  assert(mask->getType()->isIntegerTy(32) && "Expecting mask type to be i32");

  std::string fnName = "";
  switch (kind) {
  case GENX::ShflKind::XOR:
    fnName = "_Z21sub_group_shuffle_xor";
    break;
  case GENX::ShflKind::UP:
    fnName = "_Z20sub_group_shuffle_up";
    break;
  case GENX::ShflKind::DOWN:
    fnName = "_Z22sub_group_shuffle_down";
    break;
  case GENX::ShflKind::IDX:
    fnName = "_Z17sub_group_shuffle";
    break;
  }

  llvm::Type *ty = value->getType();
  if (ty->isHalfTy())
    fnName += "Dh";
  else if (ty->isFloatTy())
    fnName += "f";
  else if (ty->isDoubleTy())
    fnName += "d";
  else if (ty->isIntegerTy(8))
    fnName += "c";
  else if (ty->isIntegerTy(16))
    fnName += "s";
  else if (ty->isIntegerTy(32))
    fnName += "i";
  else if (ty->isIntegerTy(64))
    fnName += "l";
  else
    llvm_unreachable("unhandled type");

  fnName += "j";

  return createDeviceFunctionCall(builder, fnName, value->getType(),
                                  {value->getType(), mask->getType()},
                                  {value, mask});
}

// Create a call to SPIR atomic cmpxchg function.
static llvm::Value *createAtomicCmpXchg(llvm::IRBuilderBase &builder,
                                        llvm::Value *ptr, llvm::Value *cmp,
                                        llvm::Value *val) {
  assert(isa<llvm::PointerType>(ptr->getType()) && "Expecting a pointer type");
  assert(isa<llvm::IntegerType>(cmp->getType()) && "Expecting an integer type");
  assert(cmp->getType() == val->getType() && "Mismatching types");

  auto *retType = cast<llvm::IntegerType>(val->getType());
  unsigned addrSpace =
      cast<llvm::PointerType>(ptr->getType())->getAddressSpace();

  std::string fnName = "_Z12atom_cmpxchgPU";
  switch (addrSpace) {
  case mlir::GENX::GENXDialect::kGlobalMemoryAddressSpace:
    fnName += "8CLglobal";
    break;
  case mlir::GENX::GENXDialect::kSharedMemoryAddressSpace:
    fnName += "7CLlocal";
    break;
  default:
    llvm_unreachable("Unexpected address space");
  }

  switch (retType->getBitWidth()) {
  case 32:
    fnName += retType->getSignBit() ? "Viii" : "Vjjj";
    break;
  case 64:
    fnName += retType->getSignBit() ? "Vlll" : "Vmmm";
    break;
  default:
    llvm_unreachable("Unexpected bit width");
  }

  return createDeviceFunctionCall(
      builder, fnName, retType,
      {ptr->getType(), cmp->getType(), val->getType()}, {ptr, cmp, val});
}

// Create a call to SPIR atomic rmw function.
static llvm::Value *createAtomicRMW(llvm::IRBuilderBase &builder,
                                    llvm::Value *ptr, llvm::Value *val,
                                    GENX::RMWOpKind op) {
  assert(isa<llvm::PointerType>(ptr->getType()) && "Expecting a pointer type");

  auto *retType = cast<llvm::IntegerType>(val->getType());
  unsigned addrSpace =
      cast<llvm::PointerType>(ptr->getType())->getAddressSpace();

  std::string fnName = "";
  switch (op) {
  case GENX::RMWOpKind::AND:
    fnName = "_Z8atom_andPU";
    break;
  case GENX::RMWOpKind::OR:
    fnName = "_Z7atom_orPU";
    break;
  case GENX::RMWOpKind::XOR:
    fnName = "_Z8atom_xorPU";
    break;
  case GENX::RMWOpKind::ADD:
    fnName = "_Z8atom_addPU";
    break;
  case GENX::RMWOpKind::MIN:
    fnName = "_Z8atom_minPU";
    break;
  case GENX::RMWOpKind::MAX:
    fnName = "_Z8atom_maxPU";
    break;
  case GENX::RMWOpKind::XCHG:
    fnName = "_Z8atom_xchgPU";
    break;
  }

  switch (addrSpace) {
  case mlir::GENX::GENXDialect::kGlobalMemoryAddressSpace:
    fnName += "8CLglobal";
    break;
  case mlir::GENX::GENXDialect::kSharedMemoryAddressSpace:
    fnName += "7CLlocal";
    break;
  default:
    llvm_unreachable("Unexpected address space");
  }

  switch (retType->getBitWidth()) {
  case 32:
    fnName += retType->getSignBit() ? "Vii" : "Vjj";
    break;
  case 64:
    fnName += retType->getSignBit() ? "Vll" : "Vmm";
    break;
  default:
    llvm_unreachable("Unexpected bit width");
  }

  return createDeviceFunctionCall(builder, fnName, retType,
                                  {ptr->getType(), val->getType()}, {ptr, val});
}

// Create a call to GenISA_dpas for matrix multiply-add.
static llvm::Value *
createGenISADPAS(GENX::MatrixDPASOp op, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  TypeRange opTypes = op->getOperandTypes();

  llvm::Value *a = moduleTranslation.lookupValue(op.getA());
  auto aTy = llvm::FixedVectorType::get(builder.getInt16Ty(), op.getRc());
  if (a->getType() != aTy)
    a = builder.CreateBitCast(a, aTy);
  llvm::Value *b = moduleTranslation.lookupValue(op.getB());
  auto bTy = llvm::FixedVectorType::get(builder.getInt32Ty(), 8);
  if (b->getType() != bTy)
    b = builder.CreateBitCast(b, bTy);

  llvm::Function *fn = llvm::GenISAIntrinsic::getDeclaration(
      module, llvm::GenISAIntrinsic::GenISA_sub_group_dpas,
      {moduleTranslation.convertType(op->getResultTypes()[0]),
       moduleTranslation.convertType(opTypes[0]), aTy, bTy});
  SmallVector<llvm::Value *> args;
  args.push_back(moduleTranslation.lookupValue(op.getC()));
  args.push_back(a);
  args.push_back(b);

  auto int32Ty = builder.getInt32Ty();
  auto int1Ty = builder.getInt1Ty();
  args.push_back(llvm::ConstantInt::get(int32Ty, static_cast<int>(op.getPa())));
  args.push_back(llvm::ConstantInt::get(int32Ty, static_cast<int>(op.getPb())));
  args.push_back(llvm::ConstantInt::get(int32Ty, 8 /* systolic depth */));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getRc()));
  args.push_back(llvm::ConstantInt::get(int1Ty, false));
  return builder.CreateCall(fn, args);
}

// Create a call to GenISA_LSC2DBlockRead for loading a 2D submatrix.
static llvm::Value *
createGenISA2DBlockRead(GENX::Matrix2DBlockLoadOp op,
                        llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *fn = llvm::GenISAIntrinsic::getDeclaration(
      module, llvm::GenISAIntrinsic::GenISA_LSC2DBlockRead,
      {moduleTranslation.convertType(op->getResultTypes()[0])});
  assert(fn && "GenISAIntrinsic::getDeclaration() returns NULL");

  SmallVector<llvm::Value *> args(
      moduleTranslation.lookupValues(op.getOperands()));

  // The IGC intrinsic requires the first argument be int64
  assert(isa<llvm::PointerType>(args[0]->getType()) &&
         "Expecting a pointer type");
  args[0] = builder.CreatePointerCast(args[0], builder.getInt64Ty());

  auto int32Ty = builder.getInt32Ty();
  auto int1Ty = builder.getInt1Ty();
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getElemSizeInBits()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getTileWidth()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getTileHeight()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getVBlocks()));
  args.push_back(llvm::ConstantInt::get(int1Ty, op.getTranspose()));
  args.push_back(llvm::ConstantInt::get(int1Ty, op.getVnniTransform()));
  return builder.CreateCall(fn, args);
}

// Create a call to GenISA_LSC2DBlockWrite for storing to a 2D submatrix.
static void
createGenISA2DBlockWrite(GENX::Matrix2DBlockStoreOp op,
                         llvm::IRBuilderBase &builder,
                         LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  TypeRange opTypes = op->getOperandTypes();
  llvm::Function *fn = llvm::GenISAIntrinsic::getDeclaration(
      module, llvm::GenISAIntrinsic::GenISA_LSC2DBlockWrite,
      {moduleTranslation.convertType(opTypes[opTypes.size() - 1])});
  assert(fn && "GenISAIntrinsic::getDeclaration() returns NULL");

  SmallVector<llvm::Value *> args(
      moduleTranslation.lookupValues(op.getOperands()));

  // The IGC intrinsic requires the first argument be int64
  assert(isa<llvm::PointerType>(args[0]->getType()) &&
         "Expecting a pointer type");
  args[0] = builder.CreatePointerCast(args[0], builder.getInt64Ty());
  llvm::Value *lastOperand = args.pop_back_val();

  auto int32Ty = builder.getInt32Ty();
  auto int1Ty = builder.getInt1Ty();
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getElemSizeInBits()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getTileWidth()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getTileHeight()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getVBlocks()));
  args.push_back(llvm::ConstantInt::get(int1Ty, op.getTranspose()));
  args.push_back(llvm::ConstantInt::get(int1Ty, op.getVnniTransform()));
  args.push_back(lastOperand);
  builder.CreateCall(fn, args);
}

// Create a call to SPIR function for loading a joint matrix.
static llvm::Value *
createMatrixLoad(llvm::IRBuilderBase &builder, llvm::Value *res,
                 llvm::Value *ptr, llvm::Value *stride,
                 GENX::MatrixLayout layout, GENX::Scope scope,
                 std::optional<GENX::MemoryAccess> memoryAccess) {
  assert(isa<llvm::PointerType>(ptr->getType()) && "Expecting a pointer type");
  assert(false && "TODO");
  return nullptr;
}

static void createMatrixStore(llvm::IRBuilderBase &builder, llvm::Value *ptr,
                              llvm::Value *val, llvm::Value *stride,
                              GENX::MatrixLayout layout, GENX::Scope scope,
                              std::optional<GENX::MemoryAccess> memoryAccess) {
  assert(isa<llvm::PointerType>(ptr->getType()) && "Expecting a pointer type");
  assert(false && "TODO");
}

static llvm::Value *createMatrixMad(llvm::IRBuilderBase &builder,
                                    llvm::Value *a, llvm::Value *b,
                                    llvm::Value *c, GENX::Scope scope) {
  assert(false && "TODO");
  return nullptr;
}

static void createMatrixInit(llvm::IRBuilderBase &builder, llvm::Value *mat,
                             llvm::Value *val, GENX::Scope scope) {
  assert(false && "TODO");
}

static llvm::Value *createMatrixCopy(llvm::IRBuilderBase &builder,
                                     llvm::Value *res, llvm::Value *src,
                                     GENX::Scope scope) {
  assert(false && "TODO");
  return nullptr;
}

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the GENX dialect to LLVM IR.
class GENXDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/GENXConversions.inc"

    return failure();
  }

  /// Attaches metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();

    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
    StringAttr attrName = attribute.getName();
    Attribute attrVal = attribute.getValue();

    // Set calling convention for kernel
    if (attrName == GENX::GENXDialect::getKernelFuncAttrName())
      llvmFunc->setCallingConv(llvm::CallingConv::SPIR_KERNEL);

    auto attachMetadata = [&](StringRef name) {
      SmallVector<llvm::Metadata *, 3> metadata;
      llvm::Type *i64 = llvm::IntegerType::get(llvmContext, 64);
      for (int64_t i : extractFromI64ArrayAttr(attrVal)) {
        llvm::Constant *constant = llvm::ConstantInt::get(i64, i);
        metadata.push_back(llvm::ConstantAsMetadata::get(constant));
      }
      llvm::MDNode *node = llvm::MDNode::get(llvmContext, metadata);
      llvmFunc->setMetadata(name, node);
    };

    // Set max_work_group_size metadata.
    if (attrName == GENX::GENXDialect::getMaxWorkGroupSizeAttrName()) {
      if (!attrVal.dyn_cast<ArrayAttr>())
        return failure();

      attachMetadata("max_work_group_size");
    }

    // Set reqd_work_group_size metadata.
    if (attrName == GENX::GENXDialect::getReqdWorkGroupSizeAttrName()) {
      if (!attrVal.dyn_cast<ArrayAttr>())
        return failure();

      attachMetadata("reqd_work_group_size");
    }

    // Set intel_reqd_sub_group_size metadata.
    if (attrName == GENX::GENXDialect::getReqdSubGroupSizeAttrName()) {
      if (!attrVal.dyn_cast<ArrayAttr>())
        return failure();

      attachMetadata("intel_reqd_sub_group_size");
    }

    return success();
  }
};
} // namespace

void mlir::registerGENXDialectTranslation(DialectRegistry &registry) {
  registry.insert<GENX::GENXDialect>();
  registry.addExtension(+[](MLIRContext *ctx, GENX::GENXDialect *dialect) {
    dialect->addInterfaces<GENXDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerGENXDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerGENXDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
