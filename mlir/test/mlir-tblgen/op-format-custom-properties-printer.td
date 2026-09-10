// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

// A custom properties printer shadows the generated per-field helper.
// CHECK-NOT: CustomPropertiesPrinterOp::_odsPrintPropertiesAsKeyValueList
// CHECK: void CustomPropertiesPrinterOp::print
// CHECK-NOT: CustomPropertiesPrinterOp::_odsPrintPropertiesAsKeyValueList
def CustomPropertiesPrinterOp : Op<TestDialect, "custom_properties_printer"> {
  let arguments = (ins I64Attr:$attr);
  let assemblyFormat = "prop-dict attr-dict";
  let hasCustomPropertiesPrinter = 1;
  let extraClassDeclaration = [{
    void printProperties(::mlir::MLIRContext *, ::mlir::OpAsmPrinter &,
                         const Properties &,
                         ::mlir::ArrayRef<::llvm::StringRef>);
  }];
}

// Absent default-valued attributes are not printed from property storage.
// CHECK: void DefaultValuedAttrOp::_odsPrintPropertiesAsKeyValueList
// CHECK: if (shouldPrint("attr") && prop.attr) {
// CHECK-NEXT: printKey("attr", /*printValue=*/true);
def DefaultValuedAttrOp : Op<TestDialect, "default_valued_attr"> {
  let arguments = (ins DefaultValuedAttr<I64Attr, "0">:$attr);
  let assemblyFormat = "prop-dict attr-dict";
}
