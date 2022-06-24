; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s
; RUN: verify-uselistorder --opaque-pointers=0 < %s

; inalloca should roundtrip.

define void @foo(i32* inalloca(i32) %args) {
  ret void
}
; CHECK-LABEL: define void @foo(i32* inalloca(i32) %args)

define void @bar() {
  ; Use the maximum alignment, since we stuff our bit with alignment.
  %args = alloca inalloca i32, align 4294967296
  call void @foo(i32* inalloca(i32) %args)
  ret void
}
; CHECK-LABEL: define void @bar() {
; CHECK: %args = alloca inalloca i32, align 4294967296
; CHECK: call void @foo(i32* inalloca(i32) %args)
