// RUN: mlir-tblgen -gen-dialect-doc -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"
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
def AOp : Op<Test_Dialect, "a", [NoSideEffect, SingleBlockImplicitTerminator<"YieldOp">]>;

// CHECK: Dialect without a [TOC] here.
// CHECK: TOC added by tool.
// CHECK: [TOC]
// CHECK-NOT: [TOC]
// CHECK: Traits: SingleBlockImplicitTerminator<YieldOp>
// CHECK: Interfaces: NoSideEffect (MemoryEffectOpInterface)
// CHECK: Effects: MemoryEffects::Effect{}

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

// CHECK: Dialect with
// CHECK: [TOC]
// CHECK: here.
