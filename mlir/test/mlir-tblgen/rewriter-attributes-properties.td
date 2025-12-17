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
    I32:$x,
    I32Attr:$y
  );

  let results = (outs I32:$z);
}

def BOp : NS_Op<"b_op", []> {
  let arguments = (ins
    I32Attr:$y
  );

  let results = (outs I32:$z);
}

def test1 : Pat<(AOp (BOp:$x $y), $_), (AOp $x, $y)>;
// CHECK-LABEL: struct test1
// CHECK: ::llvm::LogicalResult matchAndRewrite
// CHECK-DAG: ::mlir::IntegerAttr y;
// CHECK-DAG: test::BOp x;
// CHECK-DAG: ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;
// CHECK: tblgen_ops.push_back(op0);
// CHECK: x = castedOp1;
// CHECK: tblgen_attr = castedOp1.getProperties().getY();
// CHECK: if (!(tblgen_attr))
// CHECK: y = tblgen_attr;
// CHECK: tblgen_ops.push_back(op1);

// CHECK: test::AOp tblgen_AOp_0;
// CHECK: ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
// CHECK: test::AOp::Properties tblgen_props;
// CHECK: tblgen_values.push_back((*x.getODSResults(0).begin()));
// CHECK: tblgen_props.y = ::llvm::dyn_cast_if_present<decltype(tblgen_props.y)>(y);
// CHECK: tblgen_AOp_0 = test::AOp::create(rewriter, odsLoc, tblgen_types, tblgen_values, tblgen_props);

// Note: These use strings to pick up a non-trivial storage/interface type
// difference.
def COp : NS_Op<"c_op", []> {
  let arguments = (ins
    I32:$x,
    StringProp:$y
  );

  let results = (outs I32:$z);
}

def DOp : NS_Op<"d_op", []> {
  let arguments = (ins
    StringProp:$y
  );

  let results = (outs I32:$z);
}
def test2 : Pat<(COp (DOp:$x $y), $_), (COp $x, $y)>;
// CHECK-LABEL: struct test2
// CHECK: ::llvm::LogicalResult matchAndRewrite
// CHECK-DAG: ::llvm::StringRef y;
// CHECK-DAG: test::DOp x;
// CHECK-DAG: ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;
// CHECK: tblgen_ops.push_back(op0);
// CHECK: x = castedOp1;
// CHECK: tblgen_prop = castedOp1.getProperties().getY();
// CHECK: y = tblgen_prop;
// CHECK: tblgen_ops.push_back(op1);
// CHECK: test::COp::Properties tblgen_props;
// CHECK: tblgen_props.setY(y);
