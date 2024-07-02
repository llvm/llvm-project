//===- MPIToLLVM.cpp - MPI to LLVM dialect conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MPIToLLVM/MPIToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Pass/Pass.h"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>

using namespace mlir;

namespace {

struct InitOpLowering : ConvertOpToLLVMPattern<mpi::InitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::InitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct CommRankOpLowering : ConvertOpToLLVMPattern<mpi::CommRankOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::CommRankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FinalizeOpLowering : ConvertOpToLLVMPattern<mpi::FinalizeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::FinalizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// TODO: this was copied from GPUOpsLowering.cpp:288
// is this okay, or should this be moved to some common file?
LLVM::LLVMFuncOp getOrDefineFunction(ModuleOp &moduleOp, const Location loc,
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

// TODO: this is pretty close to getOrDefineFunction, can probably be factored
LLVM::GlobalOp getOrDefineExternalStruct(ModuleOp &moduleOp, const Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         StringRef name,
                                         LLVM::LLVMStructType type) {
  LLVM::GlobalOp ret;
  if (!(ret = moduleOp.lookupSymbol<LLVM::GlobalOp>(name))) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<LLVM::GlobalOp>(
        loc, type, /*isConstant=*/false, LLVM::Linkage::External, name,
        /*value=*/Attribute(), /*alignment=*/0, 0);
  }
  return ret;
}

} // namespace

//===----------------------------------------------------------------------===//
// InitOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
InitOpLowering::matchAndRewrite(mpi::InitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
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

//===----------------------------------------------------------------------===//
// FinalizeOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
FinalizeOpLowering::matchAndRewrite(mpi::FinalizeOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  // get loc
  auto loc = op.getLoc();

  // grab a reference to the global module op:
  auto moduleOp = op->getParentOfType<ModuleOp>();

  // LLVM Function type representing `i32 MPI_Finalize()`
  auto initFuncType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {});
  // get or create function declaration:
  LLVM::LLVMFuncOp initDecl = getOrDefineFunction(moduleOp, loc, rewriter,
                                                  "MPI_Finalize", initFuncType);

  // replace init with function call
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, initDecl, ValueRange{});

  return success();
}

//===----------------------------------------------------------------------===//
// CommRankLowering
//===----------------------------------------------------------------------===//

LogicalResult
CommRankOpLowering::matchAndRewrite(mpi::CommRankOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  // get some helper vars
  auto loc = op.getLoc();
  auto context = rewriter.getContext();
  auto i32 = rewriter.getI32Type();

  // ptrType `!llvm.ptr`
  Type ptrType = LLVM::LLVMPointerType::get(context);

  // get external opaque struct pointer type
  auto commStructT = LLVM::LLVMStructType::getOpaque("MPI_ABI_Comm", context);

  // grab a reference to the global module op:
  auto moduleOp = op->getParentOfType<ModuleOp>();

  // make sure global op definition exists
  getOrDefineExternalStruct(moduleOp, loc, rewriter, "MPI_COMM_WORLD",
                            commStructT);

  // get address of @MPI_COMM_WORLD
  auto one = rewriter.create<LLVM::ConstantOp>(loc, i32, 1);
  auto rankptr = rewriter.create<LLVM::AllocaOp>(loc, ptrType, i32, one);
  auto commWorld = rewriter.create<LLVM::AddressOfOp>(
      loc, ptrType, SymbolRefAttr::get(context, "MPI_COMM_WORLD"));

  // LLVM Function type representing `i32 MPI_Comm_rank(ptr, ptr)`
  auto rankFuncType = LLVM::LLVMFunctionType::get(i32, {ptrType, ptrType});
  // get or create function declaration:
  LLVM::LLVMFuncOp initDecl = getOrDefineFunction(
      moduleOp, loc, rewriter, "MPI_Comm_rank", rankFuncType);

  // replace init with function call
  auto callOp = rewriter.create<LLVM::CallOp>(
      loc, initDecl, ValueRange{commWorld.getRes(), rankptr.getRes()});

  // load the rank into a register
  auto loadedRank =
      rewriter.create<LLVM::LoadOp>(loc, i32, rankptr.getResult());

  // if retval is checked, replace uses of retval with the results from the call
  // op
  SmallVector<Value> replacements;
  if (op.getRetval()) {
    replacements.push_back(callOp.getResult());
  }
  // replace all uses, then erase op
  replacements.push_back(loadedRank.getRes());
  rewriter.replaceOp(op, replacements);

  return success();
}

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mpi::populateMPIToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns) {
  patterns.add<InitOpLowering>(converter);
  patterns.add<CommRankOpLowering>(converter);
  patterns.add<FinalizeOpLowering>(converter);
}

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
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

void mpi::registerConvertMPIToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, mpi::MPIDialect *dialect) {
    dialect->addInterfaces<FuncToLLVMDialectInterface>();
  });
}
