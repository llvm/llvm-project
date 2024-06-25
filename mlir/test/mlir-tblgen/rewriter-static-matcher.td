// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

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

def DOp : NS_Op<"d_op", []> {
  let arguments = (ins
    Variadic<AnyInteger>:$any_integer
  );

  let results = (outs AnyInteger);
}

def Foo : NativeCodeCall<"foo($_builder, $0)">;

// Test static matcher for duplicate DagNode
// ---

// CHECK: static ::mlir::LogicalResult [[$TYPE_CONSTRAINT:__mlir_ods_local_type_constraint.*]](
// CHECK-NEXT: {{.*::mlir::Type type}}
// CHECK: static ::mlir::LogicalResult [[$ATTR_CONSTRAINT:__mlir_ods_local_attr_constraint.*]](
// CHECK-NEXT: {{.*::mlir::Attribute attr}}
// CHECK: static ::mlir::LogicalResult [[$DAG_MATCHER:static_dag_matcher.*]](
// CHECK: if(::mlir::failed([[$ATTR_CONSTRAINT]]
// CHECK: if(::mlir::failed([[$TYPE_CONSTRAINT]]

// CHECK: if(::mlir::failed([[$DAG_MATCHER]](rewriter, op1, tblgen_ops
def : Pat<(AOp (BOp I32Attr:$attr, I32:$int)),
          (AOp $int)>;

// CHECK: if(::mlir::failed([[$DAG_MATCHER]](rewriter, op1, tblgen_ops
def : Pat<(COp $_, (BOp I32Attr:$attr, I32:$int)),
          (COp $attr, $int)>;

// CHECK: auto [[$VAR:.*]] = foo(
// CHECK: ::llvm::SmallVector<::mlir::Value, 4> [[$ARR:tblgen_variadic_values_.*]];
// CHECK: [[$ARR]].push_back([[$VAR]]);
def : Pat<(AOp $x), (DOp (variadic (Foo $x)))>;
