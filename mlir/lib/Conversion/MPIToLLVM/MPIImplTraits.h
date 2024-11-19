#define MPICH_SKIP_MPICXX 1
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mpi.h>

namespace {

// when lowerring the mpi dialect to functions calls certain details
// differ between various MPI implementations. This class will provide
// these depending on the MPI implementation that got included.
struct MPIImplTraits {
  // get/create MPI_COMM_WORLD as a mlir::Value
  static mlir::Value getCommWorld(mlir::ModuleOp &moduleOp,
                                  const mlir::Location loc,
                                  mlir::ConversionPatternRewriter &rewriter);
  // get/create MPI datatype as a mlir::Value which corresponds to the given
  // mlir::Type
  static mlir::Value getDataType(const mlir::Location loc,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Type type);
};

// ****************************************************************************
// Intel MPI
#ifdef IMPI_DEVICE_EXPORT

mlir::Value
MPIImplTraits::getCommWorld(mlir::ModuleOp &moduleOp, const mlir::Location loc,
                            mlir::ConversionPatternRewriter &rewriter) {
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                 MPI_COMM_WORLD);
}

mlir::Value
MPIImplTraits::getDataType(const mlir::Location loc,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Type type) {
  int32_t mtype = 0;
  if (type.isF32())
    mtype = MPI_FLOAT;
  else if (type.isF64())
    mtype = MPI_DOUBLE;
  else if (type.isInteger(64) && !type.isUnsignedInteger())
    mtype = MPI_LONG;
  else if (type.isInteger(64))
    mtype = MPI_UNSIGNED_LONG;
  else if (type.isInteger(32) && !type.isUnsignedInteger())
    mtype = MPI_INT;
  else if (type.isInteger(32))
    mtype = MPI_UNSIGNED;
  else if (type.isInteger(16) && !type.isUnsignedInteger())
    mtype = MPI_SHORT;
  else if (type.isInteger(16))
    mtype = MPI_UNSIGNED_SHORT;
  else if (type.isInteger(8) && !type.isUnsignedInteger())
    mtype = MPI_CHAR;
  else if (type.isInteger(8))
    mtype = MPI_UNSIGNED_CHAR;
  else
    assert(false && "unsupported type");
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                 mtype);
}

// ****************************************************************************
// OpenMPI
#elif defined(OPEN_MPI) && OPEN_MPI == 1

// TODO: this is pretty close to getOrDefineFunction, can probably be factored
static mlir::LLVM::GlobalOp
getOrDefineExternalStruct(mlir::ModuleOp &moduleOp, const mlir::Location loc,
                          mlir::ConversionPatternRewriter &rewriter,
                          mlir::StringRef name,
                          mlir::LLVM::LLVMStructType type) {
  mlir::LLVM::GlobalOp ret;
  if (!(ret = moduleOp.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
    mlir::ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, type, /*isConstant=*/false, mlir::LLVM::Linkage::External, name,
        /*value=*/mlir::Attribute(), /*alignment=*/0, 0);
  }
  return ret;
}

mlir::Value
MPIImplTraits::getCommWorld(mlir::ModuleOp &moduleOp, const mlir::Location loc,
                            mlir::ConversionPatternRewriter &rewriter) {
  auto context = rewriter.getContext();
  auto i32 = rewriter.getI32Type();
  // ptrType `!llvm.ptr`
  mlir::Type ptrType = mlir::LLVM::LLVMPointerType::get(context);
  // get external opaque struct pointer type
  auto commStructT =
      mlir::LLVM::LLVMStructType::getOpaque("MPI_ABI_Comm", context);

  // make sure global op definition exists
  getOrDefineExternalStruct(moduleOp, loc, rewriter, "MPI_COMM_WORLD",
                            commStructT);

  // get address of @MPI_COMM_WORLD
  return rewriter.create<mlir::LLVM::AddressOfOp>(
      loc, ptrType, mlir::SymbolRefAttr::get(context, "MPI_COMM_WORLD"));
}

mlir::Value
MPIImplTraits::getDataType(const mlir::Location loc,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Type type) {
  assert(false && "getDataType not implemented for this MPI implementation");
  return {};
}

#else
#error "Unsupported MPI implementation"
#endif

} // namespace
