// RUN: mlir-link --sort-symbols %s %p/Inputs/byref-type-input.mlir | FileCheck %s

// CHECK-LABEL: llvm.func @bar()
// CHECK: llvm.call @foo(%1)
// CHECK: llvm.func @baz(!llvm.ptr {llvm.byref = !llvm.struct<"struct", (i32, i8)>}
// CHECK-LABEL: llvm.func @f(%arg0: !llvm.ptr {llvm.byref = !llvm.struct<"a", (i64)>})
// CHECK-LABEL: llvm.func @foo(%arg0: !llvm.ptr {llvm.byref = !llvm.struct<"struct", (i32, i8)>})
// CHECK-NEXT: llvm.call @baz(%arg0)
// CHECK-LABEL: llvm.func @g(%arg0: !llvm.ptr {llvm.byref = !llvm.struct<"a", (i64)>})

module {

  llvm.func @f(%arg0: !llvm.ptr {llvm.byref = !llvm.struct<"a", (i64)>}) {
    llvm.return
  }
  llvm.func @bar() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<"struct", (i32, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.call @foo(%1) : (!llvm.ptr) -> ()
    llvm.return
  }


  llvm.func @foo(!llvm.ptr {llvm.byref = !llvm.struct<"struct", (i32, i8)>})
}
