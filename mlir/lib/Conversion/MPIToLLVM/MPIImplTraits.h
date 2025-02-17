#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Transforms/DialectConversion.h"

// skip if no MPI C header was found
#ifdef FOUND_MPI_C_HEADER
#include <mpi.h>
#else // not FOUND_MPI_C_HEADER
#include "mpi_fallback.h"
#endif // FOUND_MPI_C_HEADER

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
  static mlir::Value getDataType(mlir::ModuleOp &moduleOp,
                                 const mlir::Location loc,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Type type);
};

// ****************************************************************************
// Intel MPI/MPICH
#if defined(IMPI_DEVICE_EXPORT) || defined(_MPI_FALLBACK_DEFS)

mlir::Value
MPIImplTraits::getCommWorld(mlir::ModuleOp &moduleOp, const mlir::Location loc,
                            mlir::ConversionPatternRewriter &rewriter) {
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                 MPI_COMM_WORLD);
}

mlir::Value
MPIImplTraits::getDataType(mlir::ModuleOp &moduleOp, const mlir::Location loc,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Type type) {
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
  // get external opaque struct pointer type
  auto commStructT =
      mlir::LLVM::LLVMStructType::getOpaque("ompi_communicator_t", context);
  const char *name = "ompi_mpi_comm_world";

  // make sure global op definition exists
  (void)getOrDefineExternalStruct(moduleOp, loc, rewriter, name, commStructT);

  // get address of symbol
  return rewriter.create<mlir::LLVM::AddressOfOp>(
      loc, mlir::LLVM::LLVMPointerType::get(context),
      mlir::SymbolRefAttr::get(context, name));
}

mlir::Value
MPIImplTraits::getDataType(mlir::ModuleOp &moduleOp, const mlir::Location loc,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Type type) {
  const char *mtype = nullptr;
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
  auto commStructT = mlir::LLVM::LLVMStructType::getOpaque(
      "ompi_predefined_datatype_t", context);
  // make sure global op definition exists
  (void)getOrDefineExternalStruct(moduleOp, loc, rewriter, mtype, commStructT);
  // get address of symbol
  return rewriter.create<mlir::LLVM::AddressOfOp>(
      loc, mlir::LLVM::LLVMPointerType::get(context),
      mlir::SymbolRefAttr::get(context, mtype));
}

#else
#error "Unsupported MPI implementation"
#endif

} // namespace
