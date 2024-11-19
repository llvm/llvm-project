//===- MPIToLLVM.cpp - MPI to LLVM dialect conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This must go first (MPI gets confused otherwise)
#include "MPIImplTraits.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MPIToLLVM/MPIToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Pass/Pass.h"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>

using namespace mlir;

// TODO: this was copied from GPUOpsLowering.cpp:288
// is this okay, or should this be moved to some common file?
static LLVM::LLVMFuncOp getOrDefineFunction(ModuleOp &moduleOp,
                                            const Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            StringRef name,
                                            LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                            LLVM::Linkage::External);
  }
  return ret;
}

namespace {

//===----------------------------------------------------------------------===//
// InitOpLowering
//===----------------------------------------------------------------------===//

struct InitOpLowering : public ConvertOpToLLVMPattern<mpi::InitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::InitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // get loc
    auto loc = op.getLoc();

    // ptrType `!llvm.ptr`
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // instantiate nullptr `%nullptr = llvm.mlir.zero : !llvm.ptr`
    auto nullPtrOp = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
    Value llvmnull = nullPtrOp.getRes();

    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // LLVM Function type representing `i32 MPI_Init(ptr, ptr)`
    auto initFuncType =
        LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {ptrType, ptrType});
    // get or create function declaration:
    LLVM::LLVMFuncOp initDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "MPI_Init", initFuncType);

    // replace init with function call
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, initDecl,
                                              ValueRange{llvmnull, llvmnull});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// FinalizeOpLowering
//===----------------------------------------------------------------------===//

struct FinalizeOpLowering : public ConvertOpToLLVMPattern<mpi::FinalizeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::FinalizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // get loc
    auto loc = op.getLoc();

    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // LLVM Function type representing `i32 MPI_Finalize()`
    auto initFuncType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {});
    // get or create function declaration:
    LLVM::LLVMFuncOp initDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "MPI_Finalize", initFuncType);

    // replace init with function call
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, initDecl, ValueRange{});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// CommRankOpLowering
//===----------------------------------------------------------------------===//

struct CommRankOpLowering : public ConvertOpToLLVMPattern<mpi::CommRankOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::CommRankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // get some helper vars
    auto loc = op.getLoc();
    auto context = rewriter.getContext();
    auto i32 = rewriter.getI32Type();

    // ptrType `!llvm.ptr`
    Type ptrType = LLVM::LLVMPointerType::get(context);

    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // get MPI_COMM_WORLD
    auto commWorld = MPIImplTraits::getCommWorld(moduleOp, loc, rewriter);

    // LLVM Function type representing `i32 MPI_Comm_rank(ptr, ptr)`
    auto rankFuncType =
        LLVM::LLVMFunctionType::get(i32, {commWorld.getType(), ptrType});
    // get or create function declaration:
    LLVM::LLVMFuncOp initDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "MPI_Comm_rank", rankFuncType);

    // replace init with function call
    auto one = rewriter.create<LLVM::ConstantOp>(loc, i32, 1);
    auto rankptr = rewriter.create<LLVM::AllocaOp>(loc, ptrType, i32, one);
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, initDecl, ValueRange{commWorld, rankptr.getRes()});

    // load the rank into a register
    auto loadedRank =
        rewriter.create<LLVM::LoadOp>(loc, i32, rankptr.getResult());

    // if retval is checked, replace uses of retval with the results from the
    // call op
    SmallVector<Value> replacements;
    if (op.getRetval()) {
      replacements.push_back(callOp.getResult());
    }
    // replace all uses, then erase op
    replacements.push_back(loadedRank.getRes());
    rewriter.replaceOp(op, replacements);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// SendOpLowering
//===----------------------------------------------------------------------===//

struct SendOpLowering : public ConvertOpToLLVMPattern<mpi::SendOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::SendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // get some helper vars
    auto loc = op.getLoc();
    auto context = rewriter.getContext();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    auto memRef = adaptor.getRef();
    auto elemType = op.getRef().getType().getElementType();

    // ptrType `!llvm.ptr`
    Type ptrType = LLVM::LLVMPointerType::get(context);

    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // get MPI_COMM_WORLD, dataType and pointer
    auto dataPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, memRef, 1)
            .getResult();
    auto offset =
        rewriter.create<LLVM::ExtractValueOp>(loc, i64, memRef, 2).getResult();
    dataPtr =
        rewriter.create<LLVM::GEPOp>(loc, ptrType, elemType, dataPtr, offset)
            .getResult();
    auto size =
        rewriter
            .create<LLVM::ExtractValueOp>(loc, memRef, ArrayRef<int64_t>{3, 0})
            .getResult();
    size = rewriter.create<LLVM::TruncOp>(loc, i32, size).getResult();
    auto dataType = MPIImplTraits::getDataType(loc, rewriter, elemType);
    auto commWorld = MPIImplTraits::getCommWorld(moduleOp, loc, rewriter);

    // LLVM Function type representing `i32 MPI_send(datatype, dst, tag, comm)`
    auto funcType = LLVM::LLVMFunctionType::get(
        i32, {ptrType, i32, i32, i32, i32, commWorld.getType()});
    // get or create function declaration:
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "MPI_Send", funcType);

    // replace op with function call
    auto funcCall = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{dataPtr, size, dataType, adaptor.getDest(), adaptor.getTag(),
                   commWorld});
    if (op.getRetval()) {
      rewriter.replaceOp(op, funcCall.getResult());
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// RecvOpLowering
//===----------------------------------------------------------------------===//

struct RecvOpLowering : public ConvertOpToLLVMPattern<mpi::RecvOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::RecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // get some helper vars
    auto loc = op.getLoc();
    auto context = rewriter.getContext();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    auto memRef = adaptor.getRef();
    auto elemType = op.getRef().getType().getElementType();

    // ptrType `!llvm.ptr`
    Type ptrType = LLVM::LLVMPointerType::get(context);

    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // get MPI_COMM_WORLD, dataType, status_ignore and pointer
    auto dataPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, memRef, 1)
            .getResult();
    auto offset =
        rewriter.create<LLVM::ExtractValueOp>(loc, i64, memRef, 2).getResult();
    dataPtr =
        rewriter.create<LLVM::GEPOp>(loc, ptrType, elemType, dataPtr, offset)
            .getResult();
    auto size =
        rewriter
            .create<LLVM::ExtractValueOp>(loc, memRef, ArrayRef<int64_t>{3, 0})
            .getResult();
    size = rewriter.create<LLVM::TruncOp>(loc, i32, size).getResult();
    auto dataType = MPIImplTraits::getDataType(loc, rewriter, elemType);
    auto commWorld = MPIImplTraits::getCommWorld(moduleOp, loc, rewriter);
    auto statusIgnore =
        rewriter
            .create<LLVM::ConstantOp>(
                loc, i64, reinterpret_cast<int64_t>(MPI_STATUS_IGNORE))
            .getResult();
    statusIgnore = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, statusIgnore)
                       .getResult();

    // LLVM Function type representing `i32 MPI_Recv(datatype, dst, tag, comm)`
    auto funcType = LLVM::LLVMFunctionType::get(
        i32, {ptrType, i32, i32, i32, i32, commWorld.getType(), ptrType});
    // get or create function declaration:
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "MPI_Recv", funcType);

    // replace op with function call
    auto funcCall = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{dataPtr, size, dataType, adaptor.getSource(),
                   adaptor.getTag(), commWorld, statusIgnore});
    if (op.getRetval()) {
      rewriter.replaceOp(op, funcCall.getResult());
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

/// Implement the interface to convert Func to LLVM.
struct FuncToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    mpi::populateMPIToLLVMConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::mpi::populateMPIToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<CommRankOpLowering, FinalizeOpLowering, InitOpLowering,
               SendOpLowering, RecvOpLowering>(converter);
}

void mlir::mpi::registerConvertMPIToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, mpi::MPIDialect *dialect) {
    dialect->addInterfaces<FuncToLLVMDialectInterface>();
  });
}
