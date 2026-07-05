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

def NonNegProp : PropConstraint<CPred<"$_self >= 0">, "non-negative integer">;

def EOp : NS_Op<"e_op", []> {
  let arguments = (ins
    I32Prop:$x,
    I64Prop:$y,
    AnyInteger:$z
  );
  let results = (outs I32:$res);
}

def FOp: NS_Op<"f_op", []> {
  let arguments = (ins
    I32Prop:$a,
    AnyInteger:$b
  );
}

// Test static matcher for duplicate DagNode
// ---

// CHECK: static ::llvm::LogicalResult [[$TYPE_CONSTRAINT:__mlir_ods_local_type_constraint.*]](
// CHECK-NEXT: {{.*::mlir::Type type}}
// CHECK: static ::llvm::LogicalResult [[$ATTR_CONSTRAINT:__mlir_ods_local_attr_constraint.*]](
// CHECK-NEXT: {{.*::mlir::Attribute attr}}
// CHECK: template <typename T>
// CHECK-NEXT: static ::llvm::LogicalResult [[$PROP_CONSTRAINT:__mlir_ods_local_prop_constraint.*]](
// CHECK-NEXT: {{.*T prop}}
// CHECK: static ::llvm::LogicalResult [[$DAG_MATCHER:static_dag_matcher.*]](
// CHECK: if(::mlir::failed([[$ATTR_CONSTRAINT]]
// CHECK: if(::mlir::failed([[$TYPE_CONSTRAINT]]
// CHECK: static ::llvm::LogicalResult [[$DAG_MATCHER2:static_dag_matcher.*]](
// CHECK-SAME: int32_t &x
// CHECK: if(::mlir::failed([[$PROP_CONSTRAINT]]
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

// CHECK: if(::mlir::failed([[$DAG_MATCHER2]]({{.*}} x{{[,)]}}
def : Pat<(AOp (EOp NonNegProp:$x, NonNegProp:$_, I32:$z)),
  (AOp $z)>;

// CHECK: if(::mlir::failed([[$DAG_MATCHER2]]({{.*}} x{{[,)]}}
def : Pat<(FOp $_, (EOp NonNegProp:$x, NonNegProp:$_, I32:$z)),
  (COp $x, $z)>;
