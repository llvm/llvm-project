// RUN: mlir-tblgen -gen-dialect-decls -I %S/../../include %s | FileCheck %s --check-prefix=DIALECT
// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=OP
// RUN: mlir-tblgen -gen-typedef-decls -I %S/../../include %s | FileCheck %s --check-prefix=TYPE

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

class AType<string name> : TypeDef<A_Dialect, name> { }


def A_TensorType : AType<"Tensor"> {
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
