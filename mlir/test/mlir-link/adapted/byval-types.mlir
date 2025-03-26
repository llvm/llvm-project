// RUN: mlir-link --sort-symbols %s %p/Inputs/byval-types-1.mlir | FileCheck %s

// CHECK: llvm.call @foo(%1)
// CHECK: llvm.func @baz(!llvm.ptr {llvm.byval = !llvm.struct<"struct", (i32, i8)>})
// CHECK: llvm.func @foo(%arg0: !llvm.ptr {llvm.byval = !llvm.struct<"struct", (i32, i8)>})
// CHECK-NEXT: llvm.call @baz(%arg0)

module {
  llvm.func @foo(!llvm.ptr {llvm.byval = !llvm.struct<"struct", (i32, i8)>})
  llvm.func @bar() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<"struct", (i32, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.call @foo(%1) : (!llvm.ptr) -> ()
    llvm.return
  }
}

