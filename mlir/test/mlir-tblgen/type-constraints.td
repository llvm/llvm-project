// RUN: mlir-tblgen -gen-type-constraint-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-type-constraint-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/CommonTypeConstraints.td"

def DummyConstraint : AnyTypeOf<[AnyInteger, Index, AnyFloat]> {
  let cppFunctionName = "isValidDummy";
}

// DECL: ::llvm::LogicalResult isValidDummy(::mlir::Type type);

// DEF: ::llvm::LogicalResult isValidDummy(::mlir::Type type) {
// DEF:   return ::llvm::success((((::llvm::isa<::mlir::IntegerType>(type))) || ((::llvm::isa<::mlir::IndexType>(type))) || ((::llvm::isa<::mlir::FloatType>(type)))));
// DEF: }
