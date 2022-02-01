// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}
class NS_Op<string mnemonic, list<Trait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def AOp : NS_Op<"a_op", []> {
  let arguments = (ins
    AnyInteger:$any_integer
  );

  let results = (outs AnyInteger);
}

def BOp : NS_Op<"b_op", []> {
  let arguments = (ins
    AnyAttr: $any_attr,
    AnyInteger
  );

  let results = (outs AnyInteger);
}

def COp : NS_Op<"c_op", []> {
  let arguments = (ins
    AnyAttr: $any_attr,
    AnyInteger
  );

  let results = (outs AnyInteger);
}

// Test static matcher for duplicate DagNode
// ---

// CHECK: static ::mlir::LogicalResult [[$TYPE_CONSTRAINT:__mlir_ods_local_type_constraint.*]](
// CHECK-NEXT: {{.*::mlir::Type type}}
// CHECK: static ::mlir::LogicalResult [[$ATTR_CONSTRAINT:__mlir_ods_local_attr_constraint.*]](
// CHECK-NEXT: {{.*::mlir::Attribute attr}}
// CHECK: static ::mlir::LogicalResult [[$DAG_MATCHER:static_dag_matcher.*]](
// CHECK: if(failed([[$ATTR_CONSTRAINT]]
// CHECK: if(failed([[$TYPE_CONSTRAINT]]

// CHECK: if(failed([[$DAG_MATCHER]](rewriter, op1, tblgen_ops
def : Pat<(AOp (BOp I32Attr:$attr, I32:$int)),
          (AOp $int)>;

// CHECK: if(failed([[$DAG_MATCHER]](rewriter, op1, tblgen_ops
def : Pat<(COp $_, (BOp I32Attr:$attr, I32:$int)),
          (COp $attr, $int)>;
