#ifndef DIALECT_LLVMIR_LLVMTYPETESTDIALECT_H_
#define DIALECT_LLVMIR_LLVMTYPETESTDIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace LLVM {
namespace {
class LLVMDialectNewTypes : public Dialect {
public:
  LLVMDialectNewTypes(MLIRContext *ctx) : Dialect(getDialectNamespace(), ctx) {
    // clang-format off
    // addTypes<LLVMVoidType,
    //          LLVMHalfType,
    //          LLVMBFloatType,
    //          LLVMFloatType,
    //          LLVMDoubleType,
    //          LLVMFP128Type,
    //          LLVMX86FP80Type,
    //          LLVMPPCFP128Type,
    //          LLVMX86MMXType,
    //          LLVMTokenType,
    //          LLVMLabelType,
    //          LLVMMetadataType,
    //          LLVMFunctionType,
    //          LLVMIntegerType,
    //          LLVMPointerType,
    //          LLVMFixedVectorType,
    //          LLVMScalableVectorType,
    //          LLVMArrayType,
    //          LLVMStructType>();
    // clang-format on
  }
  static StringRef getDialectNamespace() { return "llvm2"; }

  Type parseType(DialectAsmParser &parser) const override {
    return detail::parseType(parser);
  }
  void printType(Type type, DialectAsmPrinter &printer) const override {
    detail::printType(type.cast<LLVMType>(), printer);
  }
};
} // namespace
} // namespace LLVM

void registerLLVMTypeTestDialect() {
  mlir::registerDialect<LLVM::LLVMDialectNewTypes>();
}
} // namespace mlir

#endif // DIALECT_LLVMIR_LLVMTYPETESTDIALECT_H_
