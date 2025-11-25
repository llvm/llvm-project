; RUN: opt -passes=mergefunc -S %s | FileCheck %s
; Ensure MergeFunc handles ConstantPtrAuth correctly and does not
; merge when any ptrauth operand differs (ptr, key, int disc, addr disc).

target triple = "arm64e-apple-ios14.0.0"

declare void @baz()
@ADDR = external global i8

declare void @callee0(ptr)
declare void @callee1(ptr)
declare void @callee2(ptr)
declare void @callee3(ptr)

; different base pointer (null vs @baz)

define void @f_ptr_null() {
; CHECK-LABEL: define void @f_ptr_null()
; CHECK:       call void @callee0(ptr ptrauth (ptr null, i32 0))
entry:
  call void @callee0(ptr ptrauth (ptr null, i32 0))
  ret void
}

define void @g_ptr_baz() {
; CHECK-LABEL: define void @g_ptr_baz()
; CHECK:       call void @callee0(ptr ptrauth (ptr @baz, i32 0))
entry:
  call void @callee0(ptr ptrauth (ptr @baz, i32 0))
  ret void
}

; different key (i32 0 vs i32 1)

define void @f_key0() {
; CHECK-LABEL: define void @f_key0()
; CHECK:       call void @callee1(ptr ptrauth (ptr @baz, i32 0))
entry:
  call void @callee1(ptr ptrauth (ptr @baz, i32 0))
  ret void
}

define void @g_key1() {
; CHECK-LABEL: define void @g_key1()
; CHECK:       call void @callee1(ptr ptrauth (ptr @baz, i32 1))
entry:
  call void @callee1(ptr ptrauth (ptr @baz, i32 1))
  ret void
}

; different integer disc (i64 0 vs i64 7)

define void @f_disc0() {
; CHECK-LABEL: define void @f_disc0()
; CHECK:       call void @callee2(ptr ptrauth (ptr @baz, i32 0))
entry:
  call void @callee2(ptr ptrauth (ptr @baz, i32 0))
  ret void
}

define void @g_disc7() {
; CHECK-LABEL: define void @g_disc7()
; CHECK:       call void @callee2(ptr ptrauth (ptr @baz, i32 0, i64 7))
entry:
  call void @callee2(ptr ptrauth (ptr @baz, i32 0, i64 7))
  ret void
}

; different addr disc (ptr null vs @ADDR)

define void @f_addr_null() {
; CHECK-LABEL: define void @f_addr_null()
; CHECK:       call void @callee3(ptr ptrauth (ptr @baz, i32 0))
entry:
  call void @callee3(ptr ptrauth (ptr @baz, i32 0))
  ret void
}

define void @g_addr_ADDR() {
; CHECK-LABEL: define void @g_addr_ADDR()
; CHECK:       call void @callee3(ptr ptrauth (ptr @baz, i32 0, i64 0, ptr @ADDR))
entry:
  call void @callee3(ptr ptrauth (ptr @baz, i32 0, i64 0, ptr @ADDR))
  ret void
}

; positive test: identical ptrauth operands, should be merged

define void @merge_ptrauth_a() {
; CHECK-LABEL: define void @merge_ptrauth_a()
; CHECK:       call void @callee1(ptr ptrauth (ptr @baz, i32 0, i64 0, ptr @ADDR))
entry:
  call void @callee1(ptr ptrauth (ptr @baz, i32 0, i64 0, ptr @ADDR))
  ret void
}

define void @merge_ptrauth_b() {
; CHECK-LABEL: define void @merge_ptrauth_b()
; CHECK: tail call void @merge_ptrauth_a()
entry:
  call void @callee1(ptr ptrauth (ptr @baz, i32 0, i64 0, ptr @ADDR))
  ret void
}
