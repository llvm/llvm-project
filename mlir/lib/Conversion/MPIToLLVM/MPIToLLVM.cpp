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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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
    ret = Op::create(rewriter, loc, std::forward<Args>(args)...);
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

std::pair<Value, Value> getRawPtrAndSize(const Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         Value memRef, Type elType) {
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value dataPtr =
      LLVM::ExtractValueOp::create(rewriter, loc, ptrType, memRef, 1);
  Value offset = LLVM::ExtractValueOp::create(rewriter, loc,
                                              rewriter.getI64Type(), memRef, 2);
  Value resPtr =
      LLVM::GEPOp::create(rewriter, loc, ptrType, elType, dataPtr, offset);
  Value size;
  if (cast<LLVM::LLVMStructType>(memRef.getType()).getBody().size() > 3) {
    size = LLVM::ExtractValueOp::create(rewriter, loc, memRef,
                                        ArrayRef<int64_t>{3, 0});
    size = LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(), size);
  } else {
    size = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
  }
  return {resPtr, size};
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
  /// Different MPI implementations have different communicator types.
  /// Using i64 as a portable, intermediate type.
  /// Appropriate cast needs to take place before calling MPI functions.
  virtual Value getCommWorld(const Location loc,
                             ConversionPatternRewriter &rewriter) = 0;

  /// Type converter provides i64 type for communicator type.
  /// Converts to native type, which might be ptr or int or whatever.
  virtual Value castComm(const Location loc,
                         ConversionPatternRewriter &rewriter, Value comm) = 0;

  /// Get the MPI_STATUS_IGNORE value (typically a pointer type).
  virtual intptr_t getStatusIgnore() = 0;

  /// Get the MPI_IN_PLACE value (void *).
  virtual void *getInPlace() = 0;

  /// Gets or creates an MPI datatype as a value which corresponds to the given
  /// type.
  virtual Value getDataType(const Location loc,
                            ConversionPatternRewriter &rewriter, Type type) = 0;

  /// Gets or creates an MPI_Op value which corresponds to the given
  /// enum value.
  virtual Value getMPIOp(const Location loc,
                         ConversionPatternRewriter &rewriter,
                         mpi::MPI_ReductionOpEnum opAttr) = 0;
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
  static constexpr int MPI_MAX = 0x58000001;
  static constexpr int MPI_MIN = 0x58000002;
  static constexpr int MPI_SUM = 0x58000003;
  static constexpr int MPI_PROD = 0x58000004;
  static constexpr int MPI_LAND = 0x58000005;
  static constexpr int MPI_BAND = 0x58000006;
  static constexpr int MPI_LOR = 0x58000007;
  static constexpr int MPI_BOR = 0x58000008;
  static constexpr int MPI_LXOR = 0x58000009;
  static constexpr int MPI_BXOR = 0x5800000a;
  static constexpr int MPI_MINLOC = 0x5800000b;
  static constexpr int MPI_MAXLOC = 0x5800000c;
  static constexpr int MPI_REPLACE = 0x5800000d;
  static constexpr int MPI_NO_OP = 0x5800000e;

public:
  using MPIImplTraits::MPIImplTraits;

  ~MPICHImplTraits() override = default;

  Value getCommWorld(const Location loc,
                     ConversionPatternRewriter &rewriter) override {
    static constexpr int MPI_COMM_WORLD = 0x44000000;
    return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                    MPI_COMM_WORLD);
  }

  Value castComm(const Location loc, ConversionPatternRewriter &rewriter,
                 Value comm) override {
    return LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(), comm);
  }

  intptr_t getStatusIgnore() override { return 1; }

  void *getInPlace() override { return reinterpret_cast<void *>(-1); }

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
    return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                    mtype);
  }

  Value getMPIOp(const Location loc, ConversionPatternRewriter &rewriter,
                 mpi::MPI_ReductionOpEnum opAttr) override {
    int32_t op = MPI_NO_OP;
    switch (opAttr) {
    case mpi::MPI_ReductionOpEnum::MPI_OP_NULL:
      op = MPI_NO_OP;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_MAX:
      op = MPI_MAX;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_MIN:
      op = MPI_MIN;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_SUM:
      op = MPI_SUM;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_PROD:
      op = MPI_PROD;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_LAND:
      op = MPI_LAND;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_BAND:
      op = MPI_BAND;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_LOR:
      op = MPI_LOR;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_BOR:
      op = MPI_BOR;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_LXOR:
      op = MPI_LXOR;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_BXOR:
      op = MPI_BXOR;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_MINLOC:
      op = MPI_MINLOC;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_MAXLOC:
      op = MPI_MAXLOC;
      break;
    case mpi::MPI_ReductionOpEnum::MPI_REPLACE:
      op = MPI_REPLACE;
      break;
    }
    return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(), op);
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

  ~OMPIImplTraits() override = default;

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
    auto comm = LLVM::AddressOfOp::create(rewriter, loc,
                                          LLVM::LLVMPointerType::get(context),
                                          SymbolRefAttr::get(context, name));
    return LLVM::PtrToIntOp::create(rewriter, loc, rewriter.getI64Type(), comm);
  }

  Value castComm(const Location loc, ConversionPatternRewriter &rewriter,
                 Value comm) override {
    return LLVM::IntToPtrOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()), comm);
  }

  intptr_t getStatusIgnore() override { return 0; }

  void *getInPlace() override { return reinterpret_cast<void *>(1); }

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
    auto typeStructT =
        LLVM::LLVMStructType::getOpaque("ompi_predefined_datatype_t", context);
    // make sure global op definition exists
    getOrDefineExternalStruct(loc, rewriter, mtype, typeStructT);
    // get address of symbol
    return LLVM::AddressOfOp::create(rewriter, loc,
                                     LLVM::LLVMPointerType::get(context),
                                     SymbolRefAttr::get(context, mtype));
  }

  Value getMPIOp(const Location loc, ConversionPatternRewriter &rewriter,
                 mpi::MPI_ReductionOpEnum opAttr) override {
    StringRef op;
    switch (opAttr) {
    case mpi::MPI_ReductionOpEnum::MPI_OP_NULL:
      op = "ompi_mpi_no_op";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_MAX:
      op = "ompi_mpi_max";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_MIN:
      op = "ompi_mpi_min";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_SUM:
      op = "ompi_mpi_sum";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_PROD:
      op = "ompi_mpi_prod";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_LAND:
      op = "ompi_mpi_land";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_BAND:
      op = "ompi_mpi_band";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_LOR:
      op = "ompi_mpi_lor";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_BOR:
      op = "ompi_mpi_bor";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_LXOR:
      op = "ompi_mpi_lxor";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_BXOR:
      op = "ompi_mpi_bxor";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_MINLOC:
      op = "ompi_mpi_minloc";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_MAXLOC:
      op = "ompi_mpi_maxloc";
      break;
    case mpi::MPI_ReductionOpEnum::MPI_REPLACE:
      op = "ompi_mpi_replace";
      break;
    }
    auto context = rewriter.getContext();
    // get external opaque struct pointer type
    auto opStructT =
        LLVM::LLVMStructType::getOpaque("ompi_predefined_op_t", context);
    // make sure global op definition exists
    getOrDefineExternalStruct(loc, rewriter, op, opStructT);
    // get address of symbol
    return LLVM::AddressOfOp::create(rewriter, loc,
                                     LLVM::LLVMPointerType::get(context),
                                     SymbolRefAttr::get(context, op));
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
                           << (strAttr ? strAttr.getValue() : "<NULL>")
                           << "), defaulting to MPICH";
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
    auto nullPtrOp = LLVM::ZeroOp::create(rewriter, loc, ptrType);
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
// CommWorldOpLowering
//===----------------------------------------------------------------------===//

struct CommWorldOpLowering : public ConvertOpToLLVMPattern<mpi::CommWorldOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::CommWorldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto mpiTraits = MPIImplTraits::get(moduleOp);
    // get MPI_COMM_WORLD
    rewriter.replaceOp(op, mpiTraits->getCommWorld(op.getLoc(), rewriter));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// CommSplitOpLowering
//===----------------------------------------------------------------------===//

struct CommSplitOpLowering : public ConvertOpToLLVMPattern<mpi::CommSplitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::CommSplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto mpiTraits = MPIImplTraits::get(moduleOp);
    Type i32 = rewriter.getI32Type();
    Type ptrType = LLVM::LLVMPointerType::get(op->getContext());
    Location loc = op.getLoc();

    // get communicator
    Value comm = mpiTraits->castComm(loc, rewriter, adaptor.getComm());
    auto one = LLVM::ConstantOp::create(rewriter, loc, i32, 1);
    auto outPtr =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, comm.getType(), one);

    // int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm * newcomm)
    auto funcType =
        LLVM::LLVMFunctionType::get(i32, {comm.getType(), i32, i32, ptrType});
    // get or create function declaration:
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(moduleOp, loc, rewriter,
                                                    "MPI_Comm_split", funcType);

    auto callOp =
        LLVM::CallOp::create(rewriter, loc, funcDecl,
                             ValueRange{comm, adaptor.getColor(),
                                        adaptor.getKey(), outPtr.getRes()});

    // load the communicator into a register
    Value res = LLVM::LoadOp::create(rewriter, loc, i32, outPtr.getResult());
    res = LLVM::SExtOp::create(rewriter, loc, rewriter.getI64Type(), res);

    // if retval is checked, replace uses of retval with the results from the
    // call op
    SmallVector<Value> replacements;
    if (op.getRetval())
      replacements.push_back(callOp.getResult());

    // replace op
    replacements.push_back(res);
    rewriter.replaceOp(op, replacements);

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
    // get communicator
    Value comm = mpiTraits->castComm(loc, rewriter, adaptor.getComm());

    // LLVM Function type representing `i32 MPI_Comm_rank(ptr, ptr)`
    auto rankFuncType =
        LLVM::LLVMFunctionType::get(i32, {comm.getType(), ptrType});
    // get or create function declaration:
    LLVM::LLVMFuncOp initDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "MPI_Comm_rank", rankFuncType);

    // replace with function call
    auto one = LLVM::ConstantOp::create(rewriter, loc, i32, 1);
    auto rankptr = LLVM::AllocaOp::create(rewriter, loc, ptrType, i32, one);
    auto callOp = LLVM::CallOp::create(rewriter, loc, initDecl,
                                       ValueRange{comm, rankptr.getRes()});

    // load the rank into a register
    auto loadedRank =
        LLVM::LoadOp::create(rewriter, loc, i32, rankptr.getResult());

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
    Type elemType = op.getRef().getType().getElementType();

    // ptrType `!llvm.ptr`
    Type ptrType = LLVM::LLVMPointerType::get(context);

    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // get MPI_COMM_WORLD, dataType and pointer
    auto [dataPtr, size] =
        getRawPtrAndSize(loc, rewriter, adaptor.getRef(), elemType);
    auto mpiTraits = MPIImplTraits::get(moduleOp);
    Value dataType = mpiTraits->getDataType(loc, rewriter, elemType);
    Value comm = mpiTraits->castComm(loc, rewriter, adaptor.getComm());

    // LLVM Function type representing `i32 MPI_send(data, count, datatype, dst,
    // tag, comm)`
    auto funcType = LLVM::LLVMFunctionType::get(
        i32, {ptrType, i32, dataType.getType(), i32, i32, comm.getType()});
    // get or create function declaration:
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "MPI_Send", funcType);

    // replace op with function call
    auto funcCall = LLVM::CallOp::create(rewriter, loc, funcDecl,
                                         ValueRange{dataPtr, size, dataType,
                                                    adaptor.getDest(),
                                                    adaptor.getTag(), comm});
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
    Type elemType = op.getRef().getType().getElementType();

    // ptrType `!llvm.ptr`
    Type ptrType = LLVM::LLVMPointerType::get(context);

    // grab a reference to the global module op:
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // get MPI_COMM_WORLD, dataType, status_ignore and pointer
    auto [dataPtr, size] =
        getRawPtrAndSize(loc, rewriter, adaptor.getRef(), elemType);
    auto mpiTraits = MPIImplTraits::get(moduleOp);
    Value dataType = mpiTraits->getDataType(loc, rewriter, elemType);
    Value comm = mpiTraits->castComm(loc, rewriter, adaptor.getComm());
    Value statusIgnore = LLVM::ConstantOp::create(rewriter, loc, i64,
                                                  mpiTraits->getStatusIgnore());
    statusIgnore =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrType, statusIgnore);

    // LLVM Function type representing `i32 MPI_Recv(data, count, datatype, dst,
    // tag, comm)`
    auto funcType =
        LLVM::LLVMFunctionType::get(i32, {ptrType, i32, dataType.getType(), i32,
                                          i32, comm.getType(), ptrType});
    // get or create function declaration:
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "MPI_Recv", funcType);

    // replace op with function call
    auto funcCall = LLVM::CallOp::create(
        rewriter, loc, funcDecl,
        ValueRange{dataPtr, size, dataType, adaptor.getSource(),
                   adaptor.getTag(), comm, statusIgnore});
    if (op.getRetval())
      rewriter.replaceOp(op, funcCall.getResult());
    else
      rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// AllReduceOpLowering
//===----------------------------------------------------------------------===//

struct AllReduceOpLowering : public ConvertOpToLLVMPattern<mpi::AllReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mpi::AllReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    Type i32 = rewriter.getI32Type();
    Type i64 = rewriter.getI64Type();
    Type elemType = op.getSendbuf().getType().getElementType();

    // ptrType `!llvm.ptr`
    Type ptrType = LLVM::LLVMPointerType::get(context);
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto mpiTraits = MPIImplTraits::get(moduleOp);
    auto [sendPtr, sendSize] =
        getRawPtrAndSize(loc, rewriter, adaptor.getSendbuf(), elemType);
    auto [recvPtr, recvSize] =
        getRawPtrAndSize(loc, rewriter, adaptor.getRecvbuf(), elemType);

    // If input and output are the same, request in-place operation.
    if (adaptor.getSendbuf() == adaptor.getRecvbuf()) {
      sendPtr = LLVM::ConstantOp::create(
          rewriter, loc, i64,
          reinterpret_cast<int64_t>(mpiTraits->getInPlace()));
      sendPtr = LLVM::IntToPtrOp::create(rewriter, loc, ptrType, sendPtr);
    }

    Value dataType = mpiTraits->getDataType(loc, rewriter, elemType);
    Value mpiOp = mpiTraits->getMPIOp(loc, rewriter, op.getOp());
    Value commWorld = mpiTraits->castComm(loc, rewriter, adaptor.getComm());

    // 'int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
    //                    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)'
    auto funcType = LLVM::LLVMFunctionType::get(
        i32, {ptrType, ptrType, i32, dataType.getType(), mpiOp.getType(),
              commWorld.getType()});
    // get or create function declaration:
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "MPI_Allreduce", funcType);

    // replace op with function call
    auto funcCall = LLVM::CallOp::create(
        rewriter, loc, funcDecl,
        ValueRange{sendPtr, recvPtr, sendSize, dataType, mpiOp, commWorld});

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
  // Using i64 as a portable, intermediate type for !mpi.comm.
  // It would be nicer to somehow get the right type directly, but TLDI is not
  // available here.
  converter.addConversion([](mpi::CommType type) {
    return IntegerType::get(type.getContext(), 64);
  });
  patterns.add<CommRankOpLowering, CommSplitOpLowering, CommWorldOpLowering,
               FinalizeOpLowering, InitOpLowering, SendOpLowering,
               RecvOpLowering, AllReduceOpLowering>(converter);
}

void mpi::registerConvertMPIToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, mpi::MPIDialect *dialect) {
    dialect->addInterfaces<FuncToLLVMDialectInterface>();
  });
}
