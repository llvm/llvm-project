// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @uwtable_func() 
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @uwtable_func() attributes {uwtable_kind = #llvm.uwtableKind<sync>}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { uwtable(sync) }
