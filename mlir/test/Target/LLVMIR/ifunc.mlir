// RUN: mlir-translate -mlir-to-llvmir %s --split-input-file | FileCheck %s

// CHECK: @foo = dso_local ifunc void (ptr, i32), ptr @resolve_foo
llvm.mlir.ifunc external @foo : !llvm.func<void (ptr, i32)>, !llvm.ptr @resolve_foo {dso_local}
llvm.func @call_foo(%arg0: !llvm.ptr {llvm.noundef}, %arg1: i32 {llvm.noundef}) attributes {dso_local} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.store %arg1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
  %3 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  %4 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK: call void @foo
  llvm.call @foo(%3, %4) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> ()
  llvm.return
}
llvm.func @call_indirect_foo(%arg0: !llvm.ptr {llvm.noundef}, %arg1: i32 {llvm.noundef}) attributes {dso_local} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.addressof @foo : !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %2 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.store %arg1, %3 {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK: store ptr @foo, ptr [[STORED:%[0-9]+]]
  llvm.store %1, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
// CHECK: [[LOADED:%[0-9]+]] = load ptr, ptr [[STORED]]
  %5 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  %6 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  %7 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK: call void [[LOADED]]
  llvm.call %5(%6, %7) : !llvm.ptr, (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> ()
  llvm.return
}
llvm.func internal @resolve_foo() -> !llvm.ptr attributes {dso_local} {
  %0 = llvm.mlir.addressof @foo_1 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
llvm.func @foo_1(!llvm.ptr {llvm.noundef}, i32 {llvm.noundef})

// -----

llvm.mlir.alias external @resolver_alias : !llvm.func<ptr ()> {
  %0 = llvm.mlir.addressof @resolver : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
llvm.mlir.alias external @resolver_alias_alias : !llvm.func<ptr ()> {
  %0 = llvm.mlir.addressof @resolver_alias : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-DAG: @ifunc = ifunc float (i64), ptr @resolver_alias
// CHECK-DAG: @ifunc2 = ifunc float (i64), ptr @resolver_alias_alias
llvm.mlir.ifunc external @ifunc2 : !llvm.func<f32 (i64)>, !llvm.ptr @resolver_alias_alias
llvm.mlir.ifunc external @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver_alias
llvm.func @resolver() -> !llvm.ptr {
  %0 = llvm.mlir.constant(333 : i64) : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

// -----

// CHECK: @ifunc = linkonce_odr hidden ifunc

llvm.mlir.ifunc linkonce_odr hidden @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver {dso_local}
llvm.func @resolver() -> !llvm.ptr {
  %0 = llvm.mlir.constant(333 : i64) : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

// -----

// CHECK: @ifunc = private ifunc

llvm.mlir.ifunc private @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver {dso_local}
llvm.func @resolver() -> !llvm.ptr {
  %0 = llvm.mlir.constant(333 : i64) : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

// -----

// CHECK: @ifunc = weak ifunc

llvm.mlir.ifunc weak @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver
llvm.func @resolver() -> !llvm.ptr {
  %0 = llvm.mlir.constant(333 : i64) : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  llvm.return %1 : !llvm.ptr
}
