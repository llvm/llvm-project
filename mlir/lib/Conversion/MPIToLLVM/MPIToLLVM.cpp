//===- MPIToLLVM.cpp - MPI to LLVM dialect conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//
// Copyright (C) by Argonne National Laboratory
//    See COPYRIGHT in top-level directory
//    of MPICH source repository.
//

#include "mlir/Conversion/MPIToLLVM/MPIToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

using namespace mlir;

namespace {

template <typename Op, typename... Args>
static Op getOrDefineGlobal(ModuleOp &moduleOp, const Location loc,
                            ConversionPatternRewriter &rewriter, StringRef name,
                            Args &&...args) {
  Op ret;
  if (!(ret = moduleOp.lookupSymbol<Op>(name))) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.template create<Op>(loc, std::forward<Args>(args)...);
  }
  return ret;
}

static LLVM::LLVMFuncOp getOrDefineFunction(ModuleOp &moduleOp,
                                            const Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            StringRef name,
                                            LLVM::LLVMFunctionType type) {
  return getOrDefineGlobal<LLVM::LLVMFuncOp>(
      moduleOp, loc, rewriter, name, name, type, LLVM::Linkage::External);
}

/// When lowering the mpi dialect to functions calls certain details
/// differ between various MPI implementations. This class will provide
/// these in a generic way, depending on the MPI implementation that got
/// selected by the DLTI attribute on the module.
class MPIImplTraits {
  ModuleOp &moduleOp;

public:
  /// Instantiate a new MPIImplTraits object according to the DLTI attribute
  /// on the given module. Default to MPICH if no attribute is present or
  /// the value is unknown.
  static std::unique_ptr<MPIImplTraits> get(ModuleOp &moduleOp);

  explicit MPIImplTraits(ModuleOp &moduleOp) : moduleOp(moduleOp) {}

  virtual ~MPIImplTraits() = default;

  ModuleOp &getModuleOp() { return moduleOp; }

  /// Gets or creates MPI_COMM_WORLD as a Value.
  virtual Value getCommWorld(const Location loc,
                             ConversionPatternRewriter &rewriter) = 0;

  /// Get the MPI_STATUS_IGNORE value (typically a pointer type).
  virtual intptr_t getStatusIgnore() = 0;

  /// Gets or creates an MPI datatype as a value which corresponds to the given
  /// type.
  virtual Value getDataType(const Location loc,
                            ConversionPatternRewriter &rewriter, Type type) = 0;
};

//===----------------------------------------------------------------------===//
// Implementation details for MPICH ABI compatible MPI implementations
//===----------------------------------------------------------------------===//

class MPICHImplTraits : public MPIImplTraits {
  static constexpr int MPI_FLOAT = 0x4c00040a;
  static constexpr int MPI_DOUBLE = 0x4c00080b;
  static constexpr int MPI_INT8_T = 0x4c000137;
  static constexpr int MPI_INT16_T = 0x4c000238;
  static constexpr int MPI_INT32_T = 0x4c000439;
  static constexpr int MPI_INT64_T = 0x4c00083a;
  static constexpr int MPI_UINT8_T = 0x4c00013b;
  static constexpr int MPI_UINT16_T = 0x4c00023c;
  static constexpr int MPI_UINT32_T = 0x4c00043d;
  static constexpr int MPI_UINT64_T = 0x4c00083e;

public:
  using MPIImplTraits::MPIImplTraits;

  virtual ~MPICHImplTraits() = default;

  Value getCommWorld(const Location loc,
                     ConversionPatternRewriter &rewriter) override {
    static constexpr int MPI_COMM_WORLD = 0x44000000;
    return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                             MPI_COMM_WORLD);
  }

  intptr_t getStatusIgnore() override { return 1; }

  Value getDataType(const Location loc, ConversionPatternRewriter &rewriter,
                    Type type) override {
    int32_t mtype = 0;
    if (type.isF32())
      mtype = MPI_FLOAT;
    else if (type.isF64())
      mtype = MPI_DOUBLE;
    else if (type.isInteger(64) && !type.isUnsignedInteger())
      mtype = MPI_INT64_T;
    else if (type.isInteger(64))
      mtype = MPI_UINT64_T;
    else if (type.isInteger(32) && !type.isUnsignedInteger())
      mtype = MPI_INT32_T;
    else if (type.isInteger(32))
      mtype = MPI_UINT32_T;
    else if (type.isInteger(16) && !type.isUnsignedInteger())
      mtype = MPI_INT16_T;
    else if (type.isInteger(16))
      mtype = MPI_UINT16_T;
    else if (type.isInteger(8) && !type.isUnsignedInteger())
      mtype = MPI_INT8_T;
    else if (type.isInteger(8))
      mtype = MPI_UINT8_T;
    else
      assert(false && "unsupported type");
    return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), mtype);
  }
};

//===----------------------------------------------------------------------===//
// Implementation details for OpenMPI
//===----------------------------------------------------------------------===//
class OMPIImplTraits : public MPIImplTraits {
  LLVM::GlobalOp getOrDefineExternalStruct(const Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           StringRef name,
                                           LLVM::LLVMStructType type) {

    return getOrDefineGlobal<LLVM::GlobalOp>(
        getModuleOp(), loc, rewriter, name, type, /*isConstant=*/false,
        LLVM::Linkage::External, name,
        /*value=*/Attribute(), /*alignment=*/0, 0);
  }

public:
  using MPIImplTraits::MPIImplTraits;

  virtual ~OMPIImplTraits() = default;

  Value getCommWorld(const Location loc,
                     ConversionPatternRewriter &rewriter) override {
    auto context = rewriter.getContext();
    // get external opaque struct pointer type
    auto commStructT =
        LLVM::LLVMStructType::getOpaque("ompi_communicator_t", context);
    StringRef name = "ompi_mpi_comm_world";

    // make sure global op definition exists
    getOrDefineExternalStruct(loc, rewriter, name, commStructT);

    // get address of symbol
    return rewriter.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(context),
        SymbolRefAttr::get(context, name));
  }

  intptr_t getStatusIgnore() override { return 0; }

  Value getDataType(const Location loc, ConversionPatternRewriter &rewriter,
                    Type type) override {
    StringRef mtype;
    if (type.isF32())
      mtype = "ompi_mpi_float";
    else if (type.isF64())
      mtype = "ompi_mpi_double";
    else if (type.isInteger(64) && !type.isUnsignedInteger())
      mtype = "ompi_mpi_int64_t";
    else if (type.isInteger(64))
      mtype = "ompi_mpi_uint64_t";
    else if (type.isInteger(32) && !type.isUnsignedInteger())
      mtype = "ompi_mpi_int32_t";
    else if (type.isInteger(32))
      mtype = "ompi_mpi_uint32_t";
    else if (type.isInteger(16) && !type.isUnsignedInteger())
      mtype = "ompi_mpi_int16_t";
    else if (type.isInteger(16))
      mtype = "ompi_mpi_uint16_t";
    else if (type.isInteger(8) && !type.isUnsignedInteger())
      mtype = "ompi_mpi_int8_t";
    else if (type.isInteger(8))
      mtype = "ompi_mpi_uint8_t";
    else
      assert(false && "unsupported type");

    auto context = rewriter.getContext();
    // get external opaque struct pointer type
    auto commStructT =
        LLVM::LLVMStructType::getOpaque("ompi_predefined_datatype_t", context);
    // make sure global op definition exists
    getOrDefineExternalStruct(loc, rewriter, mtype, commStructT);
    // get address of symbol
    return rewriter.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(context),
        SymbolRefAttr::get(context, mtype));
  }
};

std::unique_ptr<MPIImplTraits> MPIImplTraits::get(ModuleOp &moduleOp) {
  auto attr = dlti::query(*&moduleOp, {"MPI:Implementation"}, true);
  if (failed(attr))
    return std::make_unique<MPICHImplTraits>(moduleOp);
  auto strAttr = dyn_cast<StringAttr>(attr.value());
  if (strAttr && strAttr.getValue() == "OpenMPI")
    return std::make_unique<OMPIImplTraits>(moduleOp);
  if (!strAttr || strAttr.getValue() != "MPICH")
    moduleOp.emitWarning() << "Unknown \"MPI:Implementation\" value in DLTI ("
                           << strAttr.getValue() << "), defaulting to MPICH";
  return std::make_unique<MPICHImplTraits>(moduleOp);
}

//===----------------------------------------------------------------------===//
// InitOpLowering
//===----------------------------------------------------------------------===//

struct InitOpLowering : public ConvertOpToLLVMPattern<mpi::InitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::InitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

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
    Location loc = op.getLoc();

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
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    Type i32 = rewriter.getI32Type();

    // ptrType `!llvm.ptr`
    Type ptrType = LLVM::LLVMPointerType::get(context);

    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto mpiTraits = MPIImplTraits::get(moduleOp);
    // get MPI_COMM_WORLD
    Value commWorld = mpiTraits->getCommWorld(loc, rewriter);

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
    if (op.getRetval())
      replacements.push_back(callOp.getResult());

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
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    Type i32 = rewriter.getI32Type();
    Type i64 = rewriter.getI64Type();
    Value memRef = adaptor.getRef();
    Type elemType = op.getRef().getType().getElementType();

    // ptrType `!llvm.ptr`
    Type ptrType = LLVM::LLVMPointerType::get(context);

    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // get MPI_COMM_WORLD, dataType and pointer
    Value dataPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, memRef, 1);
    Value offset = rewriter.create<LLVM::ExtractValueOp>(loc, i64, memRef, 2);
    dataPtr =
        rewriter.create<LLVM::GEPOp>(loc, ptrType, elemType, dataPtr, offset);
    Value size = rewriter.create<LLVM::ExtractValueOp>(loc, memRef,
                                                       ArrayRef<int64_t>{3, 0});
    size = rewriter.create<LLVM::TruncOp>(loc, i32, size);
    auto mpiTraits = MPIImplTraits::get(moduleOp);
    Value dataType = mpiTraits->getDataType(loc, rewriter, elemType);
    Value commWorld = mpiTraits->getCommWorld(loc, rewriter);

    // LLVM Function type representing `i32 MPI_send(data, count, datatype, dst,
    // tag, comm)`
    auto funcType = LLVM::LLVMFunctionType::get(
        i32, {ptrType, i32, dataType.getType(), i32, i32, commWorld.getType()});
    // get or create function declaration:
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "MPI_Send", funcType);

    // replace op with function call
    auto funcCall = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{dataPtr, size, dataType, adaptor.getDest(), adaptor.getTag(),
                   commWorld});
    if (op.getRetval())
      rewriter.replaceOp(op, funcCall.getResult());
    else
      rewriter.eraseOp(op);

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
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    Type i32 = rewriter.getI32Type();
    Type i64 = rewriter.getI64Type();
    Value memRef = adaptor.getRef();
    Type elemType = op.getRef().getType().getElementType();

    // ptrType `!llvm.ptr`
    Type ptrType = LLVM::LLVMPointerType::get(context);

    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // get MPI_COMM_WORLD, dataType, status_ignore and pointer
    Value dataPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, memRef, 1);
    Value offset = rewriter.create<LLVM::ExtractValueOp>(loc, i64, memRef, 2);
    dataPtr =
        rewriter.create<LLVM::GEPOp>(loc, ptrType, elemType, dataPtr, offset);
    Value size = rewriter.create<LLVM::ExtractValueOp>(loc, memRef,
                                                       ArrayRef<int64_t>{3, 0});
    size = rewriter.create<LLVM::TruncOp>(loc, i32, size);
    auto mpiTraits = MPIImplTraits::get(moduleOp);
    Value dataType = mpiTraits->getDataType(loc, rewriter, elemType);
    Value commWorld = mpiTraits->getCommWorld(loc, rewriter);
    Value statusIgnore = rewriter.create<LLVM::ConstantOp>(
        loc, i64, mpiTraits->getStatusIgnore());
    statusIgnore =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, statusIgnore);

    // LLVM Function type representing `i32 MPI_Recv(data, count, datatype, dst,
    // tag, comm)`
    auto funcType =
        LLVM::LLVMFunctionType::get(i32, {ptrType, i32, dataType.getType(), i32,
                                          i32, commWorld.getType(), ptrType});
    // get or create function declaration:
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "MPI_Recv", funcType);

    // replace op with function call
    auto funcCall = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{dataPtr, size, dataType, adaptor.getSource(),
                   adaptor.getTag(), commWorld, statusIgnore});
    if (op.getRetval())
      rewriter.replaceOp(op, funcCall.getResult());
    else
      rewriter.eraseOp(op);

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

void mpi::populateMPIToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns) {
  patterns.add<CommRankOpLowering, FinalizeOpLowering, InitOpLowering,
               SendOpLowering, RecvOpLowering>(converter);
}

void mpi::registerConvertMPIToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, mpi::MPIDialect *dialect) {
    dialect->addInterfaces<FuncToLLVMDialectInterface>();
  });
}
