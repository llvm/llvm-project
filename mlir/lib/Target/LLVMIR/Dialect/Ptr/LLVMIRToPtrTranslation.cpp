//===- LLVMIRToPtrTranslation.cpp - Translate LLVM IR to Ptr dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between LLVM IR and the MLIR Ptr dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/Ptr/LLVMIRToPtrTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/ModRef.h"

using namespace mlir;
using namespace mlir::ptr;

namespace {
inline ArrayRef<unsigned> getSupportedInstructionsImpl() {
  static unsigned instructions[] = {
      llvm::Instruction::AtomicCmpXchg, llvm::Instruction::AtomicRMW,
      llvm::Instruction::Load,          llvm::Instruction::Store,
      llvm::Instruction::AddrSpaceCast, llvm::Instruction::IntToPtr,
      llvm::Instruction::PtrToInt};
  return instructions;
}

inline LLVM_ATTRIBUTE_UNUSED AtomicBinOp
convertAtomicBinOpFromLLVM(::llvm::AtomicRMWInst::BinOp value) {
  switch (value) {
  case ::llvm::AtomicRMWInst::BinOp::Xchg:
    return AtomicBinOp::xchg;
  case ::llvm::AtomicRMWInst::BinOp::Add:
    return AtomicBinOp::add;
  case ::llvm::AtomicRMWInst::BinOp::Sub:
    return AtomicBinOp::sub;
  case ::llvm::AtomicRMWInst::BinOp::And:
    return AtomicBinOp::_and;
  case ::llvm::AtomicRMWInst::BinOp::Nand:
    return AtomicBinOp::nand;
  case ::llvm::AtomicRMWInst::BinOp::Or:
    return AtomicBinOp::_or;
  case ::llvm::AtomicRMWInst::BinOp::Xor:
    return AtomicBinOp::_xor;
  case ::llvm::AtomicRMWInst::BinOp::Max:
    return AtomicBinOp::max;
  case ::llvm::AtomicRMWInst::BinOp::Min:
    return AtomicBinOp::min;
  case ::llvm::AtomicRMWInst::BinOp::UMax:
    return AtomicBinOp::umax;
  case ::llvm::AtomicRMWInst::BinOp::UMin:
    return AtomicBinOp::umin;
  case ::llvm::AtomicRMWInst::BinOp::FAdd:
    return AtomicBinOp::fadd;
  case ::llvm::AtomicRMWInst::BinOp::FSub:
    return AtomicBinOp::fsub;
  case ::llvm::AtomicRMWInst::BinOp::FMax:
    return AtomicBinOp::fmax;
  case ::llvm::AtomicRMWInst::BinOp::FMin:
    return AtomicBinOp::fmin;
  case ::llvm::AtomicRMWInst::BinOp::UIncWrap:
    return AtomicBinOp::uinc_wrap;
  case ::llvm::AtomicRMWInst::BinOp::UDecWrap:
    return AtomicBinOp::udec_wrap;
  case ::llvm::AtomicRMWInst::BinOp::BAD_BINOP:
    llvm_unreachable(
        "unsupported case ::llvm::AtomicRMWInst::BinOp::BAD_BINOP");
  }
  llvm_unreachable("unknown ::llvm::AtomicRMWInst::BinOp type");
}

LLVM_ATTRIBUTE_UNUSED AtomicOrdering
convertAtomicOrderingFromLLVM(::llvm::AtomicOrdering value) {
  switch (value) {
  case ::llvm::AtomicOrdering::NotAtomic:
    return AtomicOrdering::not_atomic;
  case ::llvm::AtomicOrdering::Unordered:
    return AtomicOrdering::unordered;
  case ::llvm::AtomicOrdering::Monotonic:
    return AtomicOrdering::monotonic;
  case ::llvm::AtomicOrdering::Acquire:
    return AtomicOrdering::acquire;
  case ::llvm::AtomicOrdering::Release:
    return AtomicOrdering::release;
  case ::llvm::AtomicOrdering::AcquireRelease:
    return AtomicOrdering::acq_rel;
  case ::llvm::AtomicOrdering::SequentiallyConsistent:
    return AtomicOrdering::seq_cst;
  }
  llvm_unreachable("unknown ::llvm::AtomicOrdering type");
}

StringRef getLLVMSyncScope(llvm::Instruction *inst) {
  std::optional<llvm::SyncScope::ID> syncScopeID =
      llvm::getAtomicSyncScopeID(inst);
  if (!syncScopeID)
    return "";

  // Search the sync scope name for the given identifier. The default
  // system-level sync scope thereby maps to the empty string.
  SmallVector<StringRef> syncScopeName;
  llvm::LLVMContext &llvmContext = inst->getContext();
  llvmContext.getSyncScopeNames(syncScopeName);
  auto *it = llvm::find_if(syncScopeName, [&](StringRef name) {
    return *syncScopeID == llvmContext.getOrInsertSyncScopeID(name);
  });
  if (it != syncScopeName.end())
    return *it;
  llvm_unreachable("incorrect sync scope identifier");
}

LogicalResult convertAtomicCmpXchg(OpBuilder &builder, llvm::Instruction *inst,
                                   ArrayRef<llvm::Value *> llvmOperands,
                                   LLVM::ModuleImport &moduleImport) {
  FailureOr<Value> ptr = moduleImport.convertValue(llvmOperands[0]);
  if (failed(ptr))
    return failure();
  FailureOr<Value> cmp = moduleImport.convertValue(llvmOperands[1]);
  if (failed(cmp))
    return failure();
  FailureOr<Value> val = moduleImport.convertValue(llvmOperands[2]);
  if (failed(val))
    return failure();
  auto *cmpXchgInst = cast<llvm::AtomicCmpXchgInst>(inst);
  unsigned alignment = cmpXchgInst->getAlign().value();
  // Create the AtomicCmpXchgOp
  auto op = builder.create<AtomicCmpXchgOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), *ptr, *cmp, *val,
      convertAtomicOrderingFromLLVM(cmpXchgInst->getSuccessOrdering()),
      convertAtomicOrderingFromLLVM(cmpXchgInst->getFailureOrdering()),
      getLLVMSyncScope(cmpXchgInst), alignment, cmpXchgInst->isWeak(),
      cmpXchgInst->isVolatile());
  // Create a struct to hold the result of AtomicCmpXchgOp
  auto structType = LLVM::LLVMStructType::getLiteral(
      builder.getContext(), {op.getRes().getType(), op.getStatus().getType()},
      false);
  Value res = builder.create<LLVM::UndefOp>(op.getLoc(), structType);
  res = builder.create<LLVM::InsertValueOp>(res.getLoc(), res, op.getRes(),
                                            llvm::ArrayRef<int64_t>{0});
  res = builder.create<LLVM::InsertValueOp>(res.getLoc(), res, op.getStatus(),
                                            llvm::ArrayRef<int64_t>{1});
  moduleImport.mapOp(inst) = {res, op};
  return success();
}

LogicalResult convertAtomicRMW(OpBuilder &builder, llvm::Instruction *inst,
                               ArrayRef<llvm::Value *> llvmOperands,
                               LLVM::ModuleImport &moduleImport) {
  FailureOr<Value> ptr = moduleImport.convertValue(llvmOperands[0]);
  if (failed(ptr))
    return failure();
  FailureOr<Value> val = moduleImport.convertValue(llvmOperands[1]);
  if (failed(val))
    return failure();
  auto *atomicInst = cast<llvm::AtomicRMWInst>(inst);
  unsigned alignment = atomicInst->getAlign().value();
  // Create the AtomicRMWOp
  moduleImport.mapValue(inst) = builder.create<AtomicRMWOp>(
      moduleImport.translateLoc(inst->getDebugLoc()),
      convertAtomicBinOpFromLLVM(atomicInst->getOperation()), *ptr, *val,
      convertAtomicOrderingFromLLVM(atomicInst->getOrdering()),
      getLLVMSyncScope(atomicInst), alignment, atomicInst->isVolatile());
  return success();
}

LogicalResult convertLoad(OpBuilder &builder, llvm::Instruction *inst,
                          ArrayRef<llvm::Value *> llvmOperands,
                          LLVM::ModuleImport &moduleImport) {
  FailureOr<Value> addr = moduleImport.convertValue(llvmOperands[0]);
  if (failed(addr))
    return failure();
  auto *loadInst = cast<llvm::LoadInst>(inst);
  unsigned alignment = loadInst->getAlign().value();
  // Create the LoadOp
  moduleImport.mapValue(inst) = builder.create<LoadOp>(
      moduleImport.translateLoc(inst->getDebugLoc()),
      moduleImport.convertType(inst->getType()), *addr, alignment,
      loadInst->isVolatile(),
      loadInst->hasMetadata(llvm::LLVMContext::MD_nontemporal),
      loadInst->hasMetadata(llvm::LLVMContext::MD_invariant_load),
      convertAtomicOrderingFromLLVM(loadInst->getOrdering()),
      getLLVMSyncScope(loadInst));

  return success();
}

LogicalResult convertStore(OpBuilder &builder, llvm::Instruction *inst,
                           ArrayRef<llvm::Value *> llvmOperands,
                           LLVM::ModuleImport &moduleImport) {
  FailureOr<Value> value = moduleImport.convertValue(llvmOperands[0]);
  if (failed(value))
    return failure();
  FailureOr<Value> addr = moduleImport.convertValue(llvmOperands[1]);
  if (failed(addr))
    return failure();
  auto *storeInst = cast<llvm::StoreInst>(inst);
  unsigned alignment = storeInst->getAlign().value();
  // Create the StoreOp
  moduleImport.mapNoResultOp(inst) = builder.create<StoreOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), *value, *addr, alignment,
      storeInst->isVolatile(),
      storeInst->hasMetadata(llvm::LLVMContext::MD_nontemporal),
      convertAtomicOrderingFromLLVM(storeInst->getOrdering()),
      getLLVMSyncScope(storeInst));
  return success();
}

LogicalResult convertAddrSpaceCast(OpBuilder &builder, llvm::Instruction *inst,
                                   ArrayRef<llvm::Value *> llvmOperands,
                                   LLVM::ModuleImport &moduleImport) {
  FailureOr<Value> _llvmir_gen_operand_arg =
      moduleImport.convertValue(llvmOperands[0]);
  if (failed(_llvmir_gen_operand_arg))
    return failure();
  // Create the AddrSpaceCastOp
  moduleImport.mapValue(inst) = builder.create<AddrSpaceCastOp>(
      moduleImport.translateLoc(inst->getDebugLoc()),
      moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  return success();
}

LogicalResult convertIntToPtr(OpBuilder &builder, llvm::Instruction *inst,
                              ArrayRef<llvm::Value *> llvmOperands,
                              LLVM::ModuleImport &moduleImport) {
  FailureOr<Value> _llvmir_gen_operand_arg =
      moduleImport.convertValue(llvmOperands[0]);
  if (failed(_llvmir_gen_operand_arg))
    return failure();
  // Create the IntToPtrOp
  moduleImport.mapValue(inst) = builder.create<IntToPtrOp>(
      moduleImport.translateLoc(inst->getDebugLoc()),
      moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);

  return success();
}

LogicalResult convertPtrToInt(OpBuilder &builder, llvm::Instruction *inst,
                              ArrayRef<llvm::Value *> llvmOperands,
                              LLVM::ModuleImport &moduleImport) {
  FailureOr<Value> _llvmir_gen_operand_arg =
      moduleImport.convertValue(llvmOperands[0]);
  if (failed(_llvmir_gen_operand_arg))
    return failure();
  // Create the PtrToIntOp
  moduleImport.mapValue(inst) = builder.create<PtrToIntOp>(
      moduleImport.translateLoc(inst->getDebugLoc()),
      moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);

  return success();
}

class PtrDialectLLVMIRImportInterface : public LLVMImportDialectInterface {
public:
  using LLVMImportDialectInterface::LLVMImportDialectInterface;

  LogicalResult
  convertInstruction(OpBuilder &builder, llvm::Instruction *inst,
                     ArrayRef<llvm::Value *> llvmOperands,
                     LLVM::ModuleImport &moduleImport) const override {
    switch (inst->getOpcode()) {
    case llvm::Instruction::AtomicCmpXchg:
      return convertAtomicCmpXchg(builder, inst, llvmOperands, moduleImport);
    case llvm::Instruction::AtomicRMW:
      return convertAtomicRMW(builder, inst, llvmOperands, moduleImport);
    case llvm::Instruction::Load:
      return convertLoad(builder, inst, llvmOperands, moduleImport);
    case llvm::Instruction::Store:
      return convertStore(builder, inst, llvmOperands, moduleImport);
    case llvm::Instruction::AddrSpaceCast:
      return convertAddrSpaceCast(builder, inst, llvmOperands, moduleImport);
    case llvm::Instruction::IntToPtr:
      return convertIntToPtr(builder, inst, llvmOperands, moduleImport);
    case llvm::Instruction::PtrToInt:
      return convertPtrToInt(builder, inst, llvmOperands, moduleImport);
    default:
      break;
    }
    return failure();
  }

  ArrayRef<unsigned> getSupportedInstructions() const override {
    return getSupportedInstructionsImpl();
  }
};
} // namespace

void mlir::registerPtrDialectImport(DialectRegistry &registry) {
  registry.insert<ptr::PtrDialect>();
  registry.addExtension(+[](MLIRContext *ctx, ptr::PtrDialect *dialect) {
    dialect->addInterfaces<PtrDialectLLVMIRImportInterface>();
  });
}

void mlir::registerPtrDialectImport(MLIRContext &context) {
  DialectRegistry registry;
  registerPtrDialectImport(registry);
  context.appendDialectRegistry(registry);
}
