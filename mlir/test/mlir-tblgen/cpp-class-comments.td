// RUN: mlir-tblgen -gen-dialect-decls -I %S/../../include %s | FileCheck %s --check-prefix=DIALECT
// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=OP
// RUN: mlir-tblgen -gen-typedef-decls -I %S/../../include %s | FileCheck %s --check-prefix=TYPE
// RUN: mlir-tblgen -gen-attrdef-decls -I %S/../../include %s | FileCheck %s --check-prefix=ATTR
// RUN: mlir-tblgen -gen-attr-interface-decls -I %S/../../include %s | FileCheck %s --check-prefix=ATTR-INTERFACE
// RUN: mlir-tblgen -gen-op-interface-decls -I %S/../../include %s | FileCheck %s --check-prefix=OP-INTERFACE
// RUN: mlir-tblgen -gen-type-interface-decls -I %S/../../include %s | FileCheck %s --check-prefix=TYPE-INTERFACE
// RUN: mlir-tblgen -gen-enum-decls -I %S/../../include %s | FileCheck %s --check-prefix=ENUM

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// check dialect with summary and description
def A_Dialect : Dialect {
  let name = "a";
  let cppNamespace = "";

  let summary = "This is a summary";
  let description = [{

    This is a description, needs trimming

  }];
// DIALECT: /// This is a summary
// DIALECT-NEXT: /// This is a description, needs trimming
// DIALECT-NEXT: class ADialect : public ::mlir::Dialect {
}

def A_SomeOp1 : Op<A_Dialect, "some_op1", []>{
  let summary = "Some Op1 summary line1 \nsummary line2";

  let description = [{
      Some Op1 description
  }];

  let cppNamespace = "OP1";
// OP: namespace OP1
// OP-NEXT: /// Some Op1 summary line1
// OP-NEXT: /// summary line2
// OP-NEXT: /// Some Op1 description
// OP-NEXT: class SomeOp1;
}

// test weird characters in description
def A_SomeOp2 : Op<A_Dialect, "some_op2", []>{
  let summary = "";

  let description = [{
      $ptr (`,` $mask^)? (`,` $other^)?
      oilist(
        `a` `=` $1 | `b` `=` $2
      )
  }];
// OP: /// $ptr (`,` $mask^)? (`,` $other^)?
// OP-NEXT: /// oilist(
// OP-NEXT: /// `a` `=` $1 | `b` `=` $2
// OP-NEXT: /// )
// OP-NEXT: class SomeOp2;
}

def A_TensorType : TypeDef<A_Dialect,"Tensor"> {
  let typeName = "a.simple_a_tensor";

  let summary = "Tensor Type A summary";

  let description = [{
      Tensor Type A description
  }];

  let extraClassDeclaration = [{
    void getSignlessBlockType() const {
    }
  }];
// TYPE: /// Tensor Type A summary
// TYPE-NEXT: /// Tensor Type A description
// TYPE-NEXT: class TensorType;
}

def A_SimpleAttr : AttrDef<A_Dialect,"SimpleA"> {
  let attrName = "a.simple_attr";
  let summary = "Simple Attr A summary";

  let description = [{
      Simple Attr A description
  }];
// ATTR: /// Simple Attr A summary
// ATTR-NEXT: /// Simple Attr A description
// ATTR-NEXT: class SimpleAAttr;
}

def EncodingTrait : AttrInterface<"EncodingTrait"> {
  let cppNamespace = "mlir::a::traits";
  let description = [{
    Common trait for all layouts.
  }];
  let methods = [
  ];
// ATTR-INTERFACE: namespace mlir::a::traits {
// ATTR-INTERFACE-NEXT: /// Common trait for all layouts.
// ATTR-INTERFACE-NEXT: class EncodingTrait;
}

def SimpleEncodingTrait : AttrInterface<"SimpleEncodingTrait"> {
  let cppNamespace = "a::traits";
// ATTR-INTERFACE: namespace a::traits {
// ATTR-INTERFACE-NEXT: class SimpleEncodingTrait;
}

def SimpleOpInterface : OpInterface<"SimpleOpInterface"> {
  let cppNamespace = "a::traits";
  let description = [{

    Simple Op Interface description
    }];
// OP-INTERFACE: namespace a::traits {
// OP-INTERFACE-NEXT: /// Simple Op Interface description
// OP-INTERFACE-NEXT: class SimpleOpInterface;
}

def SimpleTypeInterface : TypeInterface<"SimpleTypeInterface"> {
  let description = [{
    Simple Type Interface description
  }];
// TYPE-INTERFACE: /// Simple Type Interface description
// TYPE-INTERFACE-NEXT: class SimpleTypeInterface;
}

def MyBitEnum: I32BitEnumAttr<"MyBitEnum", "An example bit enum",
                           [I32BitEnumCaseBit<"Bit0", 0, "tagged">,
                           I32BitEnumCaseBit<"Bit1", 1>]> {
  let genSpecializedAttr = 0;
// ENUM: // An example bit enum
// ENUM-NEXT: enum class MyBitEnum
}
