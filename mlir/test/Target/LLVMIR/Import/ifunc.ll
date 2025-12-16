; RUN: mlir-translate --import-llvm %s --split-input-file | FileCheck %s

; CHECK: llvm.mlir.ifunc external @foo : !llvm.func<void (ptr, i32)>, !llvm.ptr @resolve_foo {dso_local}
@foo = dso_local ifunc void (ptr, i32), ptr @resolve_foo

define dso_local void @call_foo(ptr noundef %0, i32 noundef %1) {
; CHECK: llvm.call @foo
  call void @foo(ptr noundef %0, i32 noundef %1)
  ret void
}

define dso_local ptr @foo_fptr() {
; CHECK: [[FPTR:%[0-9]+]] = llvm.mlir.addressof @foo
; CHECK: llvm.return [[FPTR]]
  ret ptr @foo
}

define internal ptr @resolve_foo() {
  ret ptr @foo_1
}

declare void @foo_1(ptr noundef, i32 noundef)

; // -----

define ptr @resolver() {
  ret ptr inttoptr (i64 333 to ptr)
}

@resolver_alias = alias ptr (), ptr @resolver
@resolver_alias_alias = alias ptr (), ptr @resolver_alias

; CHECK-DAG: llvm.mlir.ifunc external @ifunc : !llvm.func<f32 (i64)>, !llvm.ptr @resolver_alias
@ifunc = ifunc float (i64), ptr @resolver_alias
; CHECK-DAG: llvm.mlir.ifunc external @ifunc2 : !llvm.func<f32 (i64)>, !llvm.ptr @resolver_alias_alias
@ifunc2 = ifunc float (i64), ptr @resolver_alias_alias

; // -----

define ptr @resolver() {
  ret ptr inttoptr (i64 333 to ptr)
}

; CHECK: llvm.mlir.ifunc linkonce_odr hidden @ifunc
@ifunc = linkonce_odr hidden ifunc float (i64), ptr @resolver

; // -----

define ptr @resolver() {
  ret ptr inttoptr (i64 333 to ptr)
}

; CHECK: llvm.mlir.ifunc private @ifunc {{.*}} {dso_local}
@ifunc = private dso_local ifunc float (i64), ptr @resolver

; // -----

define ptr @resolver() {
  ret ptr inttoptr (i64 333 to ptr)
}

; CHECK: llvm.mlir.ifunc weak @ifunc
@ifunc = weak ifunc float (i64), ptr @resolver
