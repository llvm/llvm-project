// RUN: mlir-translate -mlir-to-llvmir %s --split-input-file | FileCheck %s

// CHECK: @foo = dso_local ifunc void (ptr, i32), ptr @resolve_foo
llvm.mlir.ifunc external @foo : !llvm.func<void (ptr, i32)>, !llvm.ptr @resolve_foo {dso_local}
llvm.func @call_foo(%arg0: !llvm.ptr {llvm.noundef}, %arg1: i32 {llvm.noundef}) attributes {dso_local} {
// CHECK: call void @foo
  llvm.call @foo(%arg0, %arg1) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> ()
  llvm.return
}
llvm.func @foo_fptr() -> !llvm.ptr attributes {dso_local} {
  %1 = llvm.mlir.addressof @foo : !llvm.ptr
// CHECK: ret ptr @foo
  llvm.return %1 : !llvm.ptr
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
