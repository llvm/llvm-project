// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}
class NS_Op<string mnemonic, list<OpTrait> traits> :
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
}

// Tests dag operand indexing for ops with mixed attr and operand.
// ---

def COp : NS_Op<"c_op", []> {
  let arguments = (ins
    AnyAttr: $any_attr1,
    AnyInteger,
    AnyAttr: $any_attr2,
    AnyInteger
  );
}

// Only operand 0 should be addressed during matching.
// CHECK: struct test1 : public ::mlir::RewritePattern {
// CHECK: castedOp0.getODSOperands(0).begin()).getDefiningOp()
def test1 : Pat<(BOp $attr, (AOp $input)),
                (BOp $attr, $input)>;

// Only operand 0 and 1 should be addressed during matching.
// CHECK: struct test2 : public ::mlir::RewritePattern {
// CHECK: castedOp0.getODSOperands(0);
// CHECK: castedOp0.getODSOperands(1).begin()).getDefiningOp()
def test2 : Pat<(COp $attr1, $op1, $attr2, (AOp $op2)),
                (BOp $attr1, $op2)>;


// Check rewriting with a DAG subtree in the result and remapping a location.
// CHECK: struct test3 : public ::mlir::RewritePattern {
// We expect ODSOperand 0 here, the attribute before the operand in BOp
// definition shouldn't shift the counter.
// CHECK: op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
// CHECK: rewriter.create<test::BOp>((*a.getODSResults(0).begin()).getLoc()
def test3 : Pat<(BOp $attr, (AOp:$a $input)),
                (BOp $attr, (AOp $input), (location $a))>;

