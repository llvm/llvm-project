// RUN: mlir-tblgen -gen-dialect-doc -I %S/../../include -dialect=test %s | FileCheck %s
// RUN: mlir-tblgen -gen-dialect-doc -I %S/../../include -dialect=test_toc %s | FileCheck %s --check-prefix=CHECK_TOC

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Test_Dialect : Dialect {
  let name = "test";
  let summary = "Dialect of ops to test";
  let description = [{
    Dialect without a [TOC] here.
    TOC added by tool.
  }];
  let cppNamespace = "NS";
}

def OpGroupA : OpDocGroup {
  let summary = "Group of ops";
  let description = "Grouped for some reason.";
}

let opDocGroup = OpGroupA in {
def ADOp : Op<Test_Dialect, "d", [NoMemoryEffect, SingleBlockImplicitTerminator<"YieldOp">]>;
def AAOp : Op<Test_Dialect, "a", [NoMemoryEffect, SingleBlockImplicitTerminator<"YieldOp">]>;
}

def OpGroupB : OpDocGroup {
  let summary = "Other group of ops";
  let description = "Grouped for some other reason.";
}

let opDocGroup = OpGroupB in {
def ACOp : Op<Test_Dialect, "c", [NoMemoryEffect, SingleBlockImplicitTerminator<"YieldOp">]>;
def ABOp : Op<Test_Dialect, "b", [NoMemoryEffect, SingleBlockImplicitTerminator<"YieldOp">]>;
}

def AEOp : Op<Test_Dialect, "e", [NoMemoryEffect]> {
  let summary = "Op with a summary";
  let description = "Op with a description";
  let arguments = (ins ConfinedType<AnyType, [CPred<"::llvm::isa<::mlir::TensorType>($_self)">]>:$tensor,
                       I16Attr:$int_attr);
  let results = (outs
    ConfinedType<AnyType, [CPred<"::llvm::isa<::mlir::TensorType>($_self)">]>:$output
  ); 
}

def TestAttr : DialectAttr<Test_Dialect, CPred<"true">> {
  let summary = "attribute summary";
  let description = "attribute description";
}

def TestType : DialectType<Test_Dialect, CPred<"true">> {
  let summary = "type summary";
  let description = "type description";
}

def TestAttrDef : AttrDef<Test_Dialect, "TestAttrDef"> {
  let mnemonic = "test_attr_def";
}

def TestAttrDefParams : AttrDef<Test_Dialect, "TestAttrDefParams"> {
  let mnemonic = "test_attr_def_params";
  let parameters = (ins "int":$value);

  let assemblyFormat = "`<` $value `>`";
}

def TestTypeDef : TypeDef<Test_Dialect, "TestTypeDef"> {
  let mnemonic = "test_type_def";
}

def TestTypeDefParams : TypeDef<Test_Dialect, "TestTypeDefParams"> {
  let mnemonic = "test_type_def_params";
  let parameters = (ins "int":$value);

  let assemblyFormat = "`<` $value `>`";
}

def TestEnum :
    I32EnumAttr<"TestEnum",
        "enum summary", [
        I32EnumAttrCase<"First", 0, "first">,
        I32EnumAttrCase<"Second", 1, "second">,
        I32EnumAttrCase<"Third", 2, "third">]> {
  let genSpecializedAttr = 1;
  let cppNamespace = "NS";
}

// CHECK: Dialect without a [TOC] here.
// CHECK-NEXT:  TOC added by tool.
// CHECK-EMPTY:
// CHECK-NEXT:  [TOC]
// CHECK-EMPTY:

// CHECK-NOT: [TOC]

// CHECK:       test.e
// CHECK-EMPTY:
// CHECK-NEXT:  _Op with a summary_
// CHECK-EMPTY:
// CHECK-NEXT:  Op with a description
// CHECK-EMPTY:

// CHECK: Operands:
// CHECK-EMPTY:
// CHECK-NEXT:  | Operand | Description |
// CHECK-NEXT:  | :-----: | ----------- |
// CHECK-NEXT:  | `tensor` | |
// CHECK-EMPTY:
// CHECK-NEXT:  Results:
// CHECK-EMPTY:
// CHECK-NEXT:  | Result | Description |
// CHECK-NEXT:  | :----: | ----------- |
// CHECK-NEXT:  | `output` | |
// CHECK-EMPTY:

// CHECK: Group of ops
// CHECK: test.a
// CHECK: test.d
// CHECK: Other group
// CHECK: test.b
// CHECK: test.c
// CHECK: Traits: `SingleBlockImplicitTerminator<YieldOp>`, `SingleBlock`
// CHECK: Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`
// CHECK: Effects: `MemoryEffects::Effect{}`

// CHECK:       ## Attribute constraints
// CHECK-EMPTY:
// CHECK-NEXT:  ### attribute summary
// CHECK-EMPTY:
// CHECK:       attribute description

// CHECK: TestAttrDefAttr
// CHECK: Syntax:
// CHECK: #test.test_attr_def

// CHECK: TestAttrDefParamsAttr
// CHECK: Syntax:
// CHECK: #test.test_attr_def_params

// CHECK: ## Type constraints
// CHECK: ### type summary
// CHECK: type description

// CHECK: TestTypeDefType
// CHECK: Syntax:
// CHECK: !test.test_type_def

// CHECK: TestTypeDefParamsType
// CHECK: Syntax:
// CHECK: !test.test_type_def_params

// CHECK:       ## Enums
// CHECK-EMPTY:
// CHECK-NEXT:  ### TestEnum
// CHECK-EMPTY:
// CHECK-NEXT:  _Enum summary_
// CHECK-EMPTY:
// CHECK-NEXT:  #### Cases:
// CHECK-EMPTY:
// CHECK-NEXT:  | Symbol | Value | String |
// CHECK-NEXT:  | :----: | :---: | ------ |
// CHECK-NEXT:  | First | `0` | first |
// CHECK-NEXT:  | Second | `1` | second |
// CHECK-NEXT:  | Third | `2` | third |
// CHECK-EMPTY:

def Toc_Dialect : Dialect {
  let name = "test_toc";
  let summary = "Dialect of ops to test";
  let description = [{
    Dialect with

    [TOC]

    here.
  }];
  let cppNamespace = "NS";
}
def BOp : Op<Toc_Dialect, "b", []>;

// CHECK_TOC: Dialect with
// CHECK_TOC: [TOC]
// CHECK_TOC: here.
