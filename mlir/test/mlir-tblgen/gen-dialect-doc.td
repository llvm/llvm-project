// RUN: mlir-tblgen -gen-dialect-doc -I %S/../../include -dialect=test %s | FileCheck %s
// RUN: mlir-tblgen -gen-dialect-doc -I %S/../../include -dialect=test_toc %s | FileCheck %s --check-prefix=CHECK_TOC

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
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

def AEOp : Op<Test_Dialect, "e", [NoMemoryEffect, SingleBlockImplicitTerminator<"YieldOp">]>;

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

// CHECK: Dialect without a [TOC] here.
// CHECK: TOC added by tool.
// CHECK: [TOC]

// CHECK-NOT: [TOC]
// CHECK: test.e
// CHECK: Group of ops
// CHECK: test.a
// CHECK: test.d
// CHECK: Other group
// CHECK: test.b
// CHECK: test.c
// CHECK: Traits: SingleBlockImplicitTerminator<YieldOp>
// CHECK: Interfaces: NoMemoryEffect (MemoryEffectOpInterface)
// CHECK: Effects: MemoryEffects::Effect{}

// CHECK: ## Attribute constraint definition
// CHECK: ### attribute summary
// CHECK: attribute description

// CHECK: TestAttrDefAttr
// CHECK: Syntax:
// CHECK: #test.test_attr_def

// CHECK: TestAttrDefParamsAttr
// CHECK: Syntax:
// CHECK: #test.test_attr_def_params

// CHECK: ## Type constraint definition
// CHECK: ### type summary
// CHECK: type description

// CHECK: TestTypeDefType
// CHECK: Syntax:
// CHECK: !test.test_type_def

// CHECK: TestTypeDefParamsType
// CHECK: Syntax:
// CHECK: !test.test_type_def_params

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
