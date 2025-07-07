// RUN: mlir-translate -mlir-to-llvmir %s -split-input-file | FileCheck %s

llvm.mlir.global private unnamed_addr constant @__const.main.data(dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<10 x i32>

// CHECK: @foo = dso_local ifunc void (ptr, i64), ptr @resolve_foo
llvm.mlir.ifunc @foo : !llvm.func<void (ptr, i64)>, !llvm.ptr @resolve_foo {dso_local}
llvm.func @foo_1(%arg0: !llvm.ptr {llvm.noundef}, %arg1: i64 {llvm.noundef}) attributes {dso_local} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.store %arg1, %2 {alignment = 8 : i64} : i64, !llvm.ptr
  llvm.return
}
llvm.func @foo_2(%arg0: !llvm.ptr {llvm.noundef}, %arg1: i64 {llvm.noundef}) attributes {dso_local} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.store %arg1, %2 {alignment = 8 : i64} : i64, !llvm.ptr
  llvm.return
}
llvm.func @main() -> i32 attributes {dso_local} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.addressof @__const.main.data : !llvm.ptr
  %2 = llvm.mlir.constant(40 : i64) : i64
  %3 = llvm.mlir.constant(0 : i64) : i64
  %4 = llvm.mlir.constant(10 : i64) : i64
  %5 = llvm.mlir.constant(0 : i32) : i32
  %6 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memcpy"(%6, %1, %2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %7 = llvm.getelementptr inbounds %6[%3, %3] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<10 x i32>

// CHECK: call void @foo
  llvm.call @foo(%7, %4) : (!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> ()
  llvm.return %5 : i32
}
llvm.func internal @resolve_foo() -> !llvm.ptr attributes {dso_local} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  %2 = llvm.mlir.addressof @foo_2 : !llvm.ptr
  %3 = llvm.mlir.addressof @foo_1 : !llvm.ptr
  %4 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %5 = llvm.call @check() : () -> i32
  %6 = llvm.icmp "ne" %5, %1 : i32
  llvm.cond_br %6, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  llvm.store %3, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb3
^bb2:  // pred: ^bb0
  llvm.store %2, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %7 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  llvm.return %7 : !llvm.ptr
}
llvm.func @check() -> i32

// -----

llvm.mlir.global private unnamed_addr constant @__const.main.data(dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<10 x i32>

// CHECK: @foo = dso_local ifunc void (ptr, i64), ptr @resolve_foo
llvm.mlir.ifunc @foo : !llvm.func<void (ptr, i64)>, !llvm.ptr @resolve_foo {dso_local}
llvm.func @foo_1(%arg0: !llvm.ptr {llvm.noundef}, %arg1: i64 {llvm.noundef}) attributes {dso_local} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.store %arg1, %2 {alignment = 8 : i64} : i64, !llvm.ptr
  llvm.return
}
llvm.func @foo_2(%arg0: !llvm.ptr {llvm.noundef}, %arg1: i64 {llvm.noundef}) attributes {dso_local} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.store %arg1, %2 {alignment = 8 : i64} : i64, !llvm.ptr
  llvm.return
}
llvm.func @main() -> i32 attributes {dso_local} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.addressof @__const.main.data : !llvm.ptr
  %2 = llvm.mlir.constant(40 : i64) : i64
  %3 = llvm.mlir.addressof @foo : !llvm.ptr
  %4 = llvm.mlir.constant(0 : i64) : i64
  %5 = llvm.mlir.constant(10 : i64) : i64
  %6 = llvm.mlir.constant(0 : i32) : i32
  %7 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
  %8 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memcpy"(%7, %1, %2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()

// CHECK: store ptr @foo, ptr [[STORED:%[0-9]+]]
  llvm.store %3, %8 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr

// CHECK: [[LOADED:%[0-9]+]] = load ptr, ptr [[STORED]]
  %9 = llvm.load %8 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  %10 = llvm.getelementptr inbounds %7[%4, %4] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<10 x i32>

// CHECK: call void [[LOADED]]
  llvm.call %9(%10, %5) : !llvm.ptr, (!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> ()
  llvm.return %6 : i32
}
llvm.func internal @resolve_foo() -> !llvm.ptr attributes {dso_local} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  %2 = llvm.mlir.addressof @foo_2 : !llvm.ptr
  %3 = llvm.mlir.addressof @foo_1 : !llvm.ptr
  %4 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %5 = llvm.call @check() : () -> i32
  %6 = llvm.icmp "ne" %5, %1 : i32
  llvm.cond_br %6, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  llvm.store %3, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb3
^bb2:  // pred: ^bb0
  llvm.store %2, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %7 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  llvm.return %7 : !llvm.ptr
}
llvm.func @check() -> i32
