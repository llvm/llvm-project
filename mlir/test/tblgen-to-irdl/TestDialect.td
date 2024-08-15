// RUN: tblgen-to-irdl %s -I=%S/../../include --gen-dialect-irdl-defs --dialect=test | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

// CHECK-LABEL: irdl.dialect @test {
def Test_Dialect : Dialect {
  let name = "test";
}

class Test_Type<string name, string typeMnemonic, list<Trait> traits = []>
: TypeDef<Test_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Test_Op<string mnemonic, list<Trait> traits = []>
    : Op<Test_Dialect, mnemonic, traits>;

def Test_SingletonAType : Test_Type<"SingletonAType", "singleton_a"> {}
def Test_SingletonBType : Test_Type<"SingletonBType", "singleton_b"> {}
def Test_SingletonCType : Test_Type<"SingletonCType", "singleton_c"> {}


// Check that AllOfType is converted correctly.
def Test_AndOp : Test_Op<"and"> {
  let arguments = (ins AllOfType<[Test_SingletonAType, AnyType]>:$in);
}
// CHECK-LABEL: irdl.operation @and {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.base "!test.singleton_a"
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.any
// CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.all_of(%[[v0]], %[[v1]]) 
// CHECK-NEXT:    irdl.operands(%[[v2]])
// CHECK-NEXT:    irdl.results()
// CHECK-NEXT:  }


// Check that AnyType is converted correctly.
def Test_AnyOp : Test_Op<"any"> {
  let arguments = (ins AnyType:$in);
}
// CHECK-LABEL: irdl.operation @any {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.any
// CHECK-NEXT:    irdl.operands(%[[v0]])
// CHECK-NEXT:    irdl.results()
// CHECK-NEXT:  }


// Check that AnyTypeOf is converted correctly.
def Test_OrOp : Test_Op<"or"> {
  let arguments = (ins AnyTypeOf<[Test_SingletonAType, Test_SingletonBType, Test_SingletonCType]>:$in);
}
// CHECK-LABEL: irdl.operation @or {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.base "!test.singleton_a"
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.base "!test.singleton_b"
// CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.base "!test.singleton_c"
// CHECK-NEXT:    %[[v3:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]], %[[v2]]) 
// CHECK-NEXT:    irdl.operands(%[[v3]])
// CHECK-NEXT:    irdl.results()
// CHECK-NEXT:  }


// Check that variadics and optionals are converted correctly.
def Test_VariadicityOp : Test_Op<"variadicity"> {
  let arguments = (ins Variadic<Test_SingletonAType>:$variadic,
                       Optional<Test_SingletonBType>:$optional,
                       Test_SingletonCType:$required);
}
// CHECK-LABEL: irdl.operation @variadicity {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.base "!test.singleton_a"
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.base "!test.singleton_b"
// CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.base "!test.singleton_c"
// CHECK-NEXT:    irdl.operands(variadic %[[v0]], optional %[[v1]], %[[v2]])
// CHECK-NEXT:    irdl.results()
// CHECK-NEXT:  }
