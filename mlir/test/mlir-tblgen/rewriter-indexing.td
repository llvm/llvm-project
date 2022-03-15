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

def DOp : NS_Op<"d_op", []> {
  let arguments = (ins
    AnyInteger:$v1,
    AnyInteger:$v2,
    AnyInteger:$v3,
    AnyInteger:$v4,
    AnyInteger:$v5,
    AnyInteger:$v6,
    AnyInteger:$v7,
    AnyInteger:$v8,
    AnyInteger:$v9,
    AnyInteger:$v10
  );

  let results = (outs AnyInteger);
}

def NativeBuilder :
  NativeCodeCall<[{
    nativeCall($_builder, $_loc, $0, $1, $2, $3, $4, $5, $6, $7, $8, $9)
  }]>;

// Check Pattern with large number of DAG arguments passed to NativeCodeCall
// CHECK: struct test4 : public ::mlir::RewritePattern {
// CHECK: nativeCall(rewriter, odsLoc, (*v1.begin()), (*v2.begin()), (*v3.begin()), (*v4.begin()), (*v5.begin()), (*v6.begin()), (*v7.begin()), (*v8.begin()), (*v9.begin()), (*v10.begin()))
def test4 : Pat<(DOp $v1, $v2, $v3, $v4, $v5, $v6, $v7, $v8, $v9, $v10),
                (NativeBuilder $v1, $v2, $v3, $v4, $v5, $v6, $v7, $v8, $v9, $v10)>;

// CHECK: struct test5 : public ::mlir::RewritePattern {
// CHECK: foo(rewriter, (*v4.begin()), (*v5.begin()), (*v6.begin()), (*v7.begin()), (*v8.begin()), (*v9.begin()), (*v10.begin()))
def test5 : Pat<(DOp $v1, $v2, $v3, $v4, $v5, $v6, $v7, $v8, $v9, $v10),
                (NativeCodeCall<[{ foo($_builder, $3...) }]> $v1, $v2, $v3, $v4, $v5, $v6, $v7, $v8, $v9, $v10)>;

// Check Pattern with return type builder.
def SameTypeAs : NativeCodeCall<"$0.getType()">;
// CHECK: struct test6 : public ::mlir::RewritePattern {
// CHECK: tblgen_types.push_back((*v2.begin()).getType())
// CHECK: tblgen_types.push_back(rewriter.getI32Type())
// CHECK: nativeVar_1 = doSomething((*v3.begin()))
// CHECK: tblgen_types.push_back(nativeVar_1)
def test6 : Pat<(DOp $v1, $v2, $v3, $v4, $v5, $v6, $v7, $v8, $v9, $v10),
                (AOp (AOp $v1, (returnType $v2, "$_builder.getI32Type()", (NativeCodeCall<"doSomething($0)"> $v3))))>;
