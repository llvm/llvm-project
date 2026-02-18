; RUN: llvm-as < %s | llvm-dis --materialize-metadata --show-annotations | FileCheck %s

; CHECK: @global_var = global i32 1
; CHECK: @alias = alias i32, ptr @global_var
; CHECK: @ifunc = ifunc i32 (), ptr @ifunc_resolver
@global_var = global i32 1
@alias = alias i32, ptr @global_var
@ifunc = ifunc i32 (), ptr @ifunc_resolver

; CHECK: ; Materializable
; CHECK-NEXT: define ptr @ifunc_resolver() {}
define ptr @ifunc_resolver() {
  ret ptr @defined_function
}

; CHECK: ; Materializable
; CHECK-NEXT: define void @defined_function() {}
define void @defined_function() {
  ret void
}

; CHECK: declare void @declared_function()
declare void @declared_function()
