// RUN: mlir-tblgen -gen-attr-constraint-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-attr-constraint-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/CommonAttrConstraints.td"

def DummyConstraint : AnyAttrOf<[APIntAttr, ArrayAttr, UnitAttr]> {
  let cppFunctionName = "isValidDummy";
}

// DECL: bool isValidDummy(::mlir::Attribute attr);

// DEF: bool isValidDummy(::mlir::Attribute attr) {
// DEF:   return (((::llvm::isa<::mlir::IntegerAttr>(attr))) || ((::llvm::isa<::mlir::ArrayAttr>(attr))) || ((::llvm::isa<::mlir::UnitAttr>(attr))));
// DEF: }
