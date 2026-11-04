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

class Test_Attr<string name, string attrMnemonic> : AttrDef<Test_Dialect, name> {
  let mnemonic = attrMnemonic;
}

class Test_Op<string mnemonic, list<Trait> traits = []>
    : Op<Test_Dialect, mnemonic, traits>;

// CHECK: irdl.type @"!singleton_a"
def Test_SingletonAType : Test_Type<"SingletonAType", "singleton_a"> {}
// CHECK: irdl.type @"!singleton_b"
def Test_SingletonBType : Test_Type<"SingletonBType", "singleton_b"> {}
// CHECK: irdl.type @"!singleton_c"
def Test_SingletonCType : Test_Type<"SingletonCType", "singleton_c"> {}
// CHECK: irdl.attribute @"#test"
def Test_TestAttr : Test_Attr<"Test", "test"> {}


// Check that AllOfType is converted correctly.
def Test_AndOp : Test_Op<"and"> {
  let arguments = (ins AllOfType<[Test_SingletonAType, AnyType]>:$in);
}
// CHECK-LABEL: irdl.operation @and {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.base @test::@"!singleton_a"
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.any
// CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.all_of(%[[v0]], %[[v1]])
// CHECK-NEXT:    irdl.operands(in: %[[v2]])
// CHECK-NEXT:  }


// Check that AnyType is converted correctly.
def Test_AnyOp : Test_Op<"any"> {
  let arguments = (ins AnyType:$in);
}
// CHECK-LABEL: irdl.operation @any {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.any
// CHECK-NEXT:    irdl.operands(in: %[[v0]])
// CHECK-NEXT:  }

// Check attributes are converted correctly.
def Test_AttributesOp : Test_Op<"attributes"> {
  let arguments = (ins I16Attr:$int_attr,
                       Test_TestAttr:$test_attr);
}
// CHECK-LABEL: irdl.operation @attributes {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.base "!builtin.integer"
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.base @test::@"#test"
// CHECK-NEXT:    irdl.attributes {"int_attr" = %[[v0]], "test_attr" = %[[v1]]}
// CHECK-NEXT:  }

// Check confined types are converted correctly.
def Test_ConfinedOp : Test_Op<"confined"> {
  let arguments = (ins ConfinedType<AnyType, [CPred<"::llvm::isa<::mlir::TensorType>($_self)">]>:$tensor,
                       ConfinedType<AnyType, [And<[CPred<"::llvm::isa<::mlir::VectorType>($_self)">
                                             , CPred<"::llvm::cast<::mlir::VectorType>($_self).getRank() > 0">]>]>:$vector);
}
// CHECK-LABEL: irdl.operation @confined {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.any
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.c_pred "(::llvm::isa<::mlir::TensorType>($_self))"
// CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.all_of(%[[v0]], %[[v1]])
// CHECK-NEXT:    %[[v3:[^ ]*]] = irdl.any
// CHECK-NEXT:    %[[v4:[^ ]*]] = irdl.c_pred "(::llvm::isa<::mlir::VectorType>($_self))"
// CHECK-NEXT:    %[[v5:[^ ]*]] = irdl.c_pred "(::llvm::cast<::mlir::VectorType>($_self).getRank() > 0)"
// CHECK-NEXT:    %[[v6:[^ ]*]] = irdl.all_of(%[[v4]], %[[v5]])
// CHECK-NEXT:    %[[v7:[^ ]*]] = irdl.all_of(%[[v3]], %[[v6]])
// CHECK-NEXT:    irdl.operands(tensor: %[[v2]], vector: %[[v7]])
// CHECK-NEXT:  }

// Check generic integer types are converted correctly.
def Test_Integers : Test_Op<"integers"> {
  let arguments = (ins AnyI8:$any_int,
                       AnyInteger:$any_integer);
}
// CHECK-LABEL: irdl.operation @integers {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i8
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is si8
// CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is ui8
// CHECK-NEXT:    %[[v3:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]], %[[v2]])
// CHECK-NEXT:    %[[v4:[^ ]*]] = irdl.base "!builtin.integer"
// CHECK-NEXT:    irdl.operands(any_int: %[[v3]], any_integer: %[[v4]])
// CHECK-NEXT:  }

// Check that IRDL names are properly generated when needed.
def Test_NamesOp : Test_Op<"names"> {
  let arguments = (ins I32:$a,
                       I32:$hello,
                       I32:$unnamed0,
                       I32);
  let regions = (region AnyRegion:$unnamed1);
  let results = (outs I32:$unnamed2);
}
// CHECK-LABEL: irdl.operation @names {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i32
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32
// CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i32
// CHECK-NEXT:    %[[v3:[^ ]*]] = irdl.is i32
// CHECK-NEXT:    %[[v4:[^ ]*]] = irdl.is i32
// CHECK-NEXT:    %[[reg:[^ ]*]] = irdl.region
// CHECK-NEXT:    irdl.operands(a: %[[v0]], hello: %[[v1]], unnamed0: %[[v2]], unnamed3: %[[v3]])
// CHECK-NEXT:    irdl.results(unnamed2: %[[v4]])
// CHECK-NEXT:    irdl.regions(unnamed1: %[[reg]])
// CHECK-NEXT:  }

// Check that AnyTypeOf is converted correctly.
def Test_OrOp : Test_Op<"or"> {
  let arguments = (ins AnyTypeOf<[Test_SingletonAType, Test_SingletonBType, Test_SingletonCType]>:$in);
}
// CHECK-LABEL: irdl.operation @or {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.base @test::@"!singleton_a"
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.base @test::@"!singleton_b"
// CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.base @test::@"!singleton_c"
// CHECK-NEXT:    %[[v3:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]], %[[v2]])
// CHECK-NEXT:    irdl.operands(in: %[[v3]])
// CHECK-NEXT:  }

// Check regions are converted correctly.
def Test_RegionsOp : Test_Op<"regions"> {
  let regions = (region AnyRegion:$any_region,
                        SizedRegion<1>:$single_block_region);
}
// CHECK-LABEL: irdl.operation @regions {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.region
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.region with size 1
// CHECK-NEXT:    irdl.regions(any_region: %[[v0]], single_block_region: %[[v1]])
// CHECK-NEXT:  }

// Check that various types are converted correctly.
def Test_TypesOp : Test_Op<"types"> {
  let arguments = (ins I32:$a,
                       SI64:$b,
                       UI8:$c,
                       Index:$d,
                       F32:$e,
                       NoneType:$f,
                       Complex<F8E4M3FN>);
}
// CHECK-LABEL: irdl.operation @types {
// CHECK-NEXT:    %{{.*}} = irdl.is i32
// CHECK-NEXT:    %{{.*}} = irdl.is si64
// CHECK-NEXT:    %{{.*}} = irdl.is ui8
// CHECK-NEXT:    %{{.*}} = irdl.is index
// CHECK-NEXT:    %{{.*}} = irdl.is f32
// CHECK-NEXT:    %{{.*}} = irdl.is none
// CHECK-NEXT:    %{{.*}} = irdl.is complex<f8E4M3FN>
// CHECK-NEXT:    irdl.operands({{.*}})
// CHECK-NEXT:  }

// Check that variadics and optionals are converted correctly.
def Test_VariadicityOp : Test_Op<"variadicity"> {
  let arguments = (ins Variadic<Test_SingletonAType>:$variadic,
                       Optional<Test_SingletonBType>:$optional,
                       Test_SingletonCType:$required);
}
// CHECK-LABEL: irdl.operation @variadicity {
// CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.base @test::@"!singleton_a"
// CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.base @test::@"!singleton_b"
// CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.base @test::@"!singleton_c"
// CHECK-NEXT:    irdl.operands(variadic: variadic %[[v0]], optional: optional %[[v1]], required: %[[v2]])
// CHECK-NEXT:  }
