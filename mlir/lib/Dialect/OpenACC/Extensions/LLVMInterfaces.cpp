//===- LLVMInterfaces.cpp - OpenACC external models -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Extensions/LLVMInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::acc;

namespace {
struct LLVMPointerPointerLikeModel
    : public PointerLikeType::ExternalModel<LLVMPointerPointerLikeModel,
                                            LLVM::LLVMPointerType> {
  Type getElementType(Type pointer) const { return Type(); }

  mlir::Value genLoad(Type pointer, OpBuilder &builder, Location loc,
                      TypedValue<PointerLikeType> srcPtr,
                      Type valueType) const {
    // For LLVM pointers, we need the valueType to determine what to load
    if (!valueType)
      return {};

    return LLVM::LoadOp::create(builder, loc, valueType, srcPtr);
  }

  bool genStore(Type pointer, OpBuilder &builder, Location loc,
                Value valueToStore, TypedValue<PointerLikeType> destPtr) const {
    LLVM::StoreOp::create(builder, loc, valueToStore, destPtr);
    return true;
  }

  Value genCast(Type, OpBuilder &builder, Location loc, Value value,
                Type resultType) const {
    if (value.getType() == resultType)
      return value;

    auto srcPtrTy = dyn_cast<LLVM::LLVMPointerType>(value.getType());
    auto dstPtrTy = dyn_cast<LLVM::LLVMPointerType>(resultType);
    if (srcPtrTy && dstPtrTy) {
      if (srcPtrTy.getAddressSpace() != dstPtrTy.getAddressSpace())
        return LLVM::AddrSpaceCastOp::create(builder, loc, resultType, value);
      return value;
    }

    if (srcPtrTy && isa<IntegerType>(resultType))
      return LLVM::PtrToIntOp::create(builder, loc, resultType, value);

    if (dstPtrTy) {
      Value intVal = value;
      if (isa<IndexType>(value.getType()))
        intVal = arith::IndexCastUIOp::create(builder, loc,
                                              builder.getI64Type(), value);
      if (isa<IntegerType>(intVal.getType()))
        return LLVM::IntToPtrOp::create(builder, loc, resultType, intVal);
    }

    if (auto resPtrLike = dyn_cast<PointerLikeType>(resultType))
      if (!isa<LLVM::LLVMPointerType>(resPtrLike))
        if (Value v = resPtrLike.genCast(builder, loc, value, resultType))
          return v;
    if (auto valPtrLike = dyn_cast<PointerLikeType>(value.getType()))
      if (!isa<LLVM::LLVMPointerType>(valPtrLike))
        if (Value v = valPtrLike.genCast(builder, loc, value, resultType))
          return v;

    return UnrealizedConversionCastOp::create(builder, loc,
                                              TypeRange(resultType), value)
        .getResult(0);
  }
};

struct LLVMAddressOfGlobalModel
    : public AddressOfGlobalOpInterface::ExternalModel<LLVMAddressOfGlobalModel,
                                                       LLVM::AddressOfOp> {
  SymbolRefAttr getSymbol(Operation *op) const {
    auto addressOfOp = cast<LLVM::AddressOfOp>(op);
    return addressOfOp.getGlobalNameAttr();
  }
};
} // namespace

void mlir::acc::registerLLVMInterfaces(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, OpenACCDialect *) {
    ctx->getOrLoadDialect<LLVM::LLVMDialect>();
    ctx->getOrLoadDialect<arith::ArithDialect>();
    LLVM::LLVMPointerType::attachInterface<LLVMPointerPointerLikeModel>(*ctx);
    LLVM::AddressOfOp::attachInterface<LLVMAddressOfGlobalModel>(*ctx);
  });
}
