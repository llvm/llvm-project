; RUN: opt -passes=mergefunc -S %s | FileCheck %s
; Ensure MergeFunc handles ConstantPtrAuth correctly and does not
; merge when any ptrauth operand differs (ptr, key, int disc, addr disc).

target triple = "arm64e-apple-ios14.0.0"

declare void @baz()
@ADDR = external global i8

declare void @sink0()
declare void @sink1()
declare void @sink2()
declare void @sink3()

; different base pointer (null vs @baz)

define void @f_ptr_null() {
; CHECK-LABEL: define void @f_ptr_null()
; CHECK:       call void @sink0()
; CHECK:       call void ptrauth (ptr null, i32 0)(
entry:
  call void @sink0()
  call void ptrauth (ptr null, i32 0)(ptr null)
  ret void
}

define void @g_ptr_baz() {
; CHECK-LABEL: define void @g_ptr_baz()
; CHECK:       call void @sink0()
; CHECK:       call void ptrauth (ptr @baz, i32 0)(
entry:
  call void @sink0()
  call void ptrauth (ptr @baz, i32 0)(ptr null)
  ret void
}

; different key (i32 0 vs i32 1)

define void @f_key0() {
; CHECK-LABEL: define void @f_key0()
; CHECK:       call void @sink1()
; CHECK:       call void ptrauth (ptr @baz, i32 0)(
entry:
  call void @sink1()
  call void ptrauth (ptr @baz, i32 0)(ptr null)
  ret void
}

define void @g_key1() {
; CHECK-LABEL: define void @g_key1()
; CHECK:       call void @sink1()
; CHECK:       call void ptrauth (ptr @baz, i32 1)(
entry:
  call void @sink1()
  call void ptrauth (ptr @baz, i32 1)(ptr null)
  ret void
}

; different integer disc (i64 0 vs i64 7)

define void @f_disc0() {
; CHECK-LABEL: define void @f_disc0()
; CHECK:       call void @sink2()
; CHECK:       call void ptrauth (ptr @baz, i32 0)(
entry:
  call void @sink2()
  call void ptrauth (ptr @baz, i32 0)(ptr null)
  ret void
}

define void @g_disc7() {
; CHECK-LABEL: define void @g_disc7()
; CHECK:       call void @sink2()
; CHECK:       call void ptrauth (ptr @baz, i32 0, i64 7)(
entry:
  call void @sink2()
  call void ptrauth (ptr @baz, i32 0, i64 7)(ptr null)
  ret void
}

; different addr disc (ptr null vs @ADDR)

define void @f_addr_null() {
; CHECK-LABEL: define void @f_addr_null()
; CHECK:       call void @sink3()
; CHECK:       call void ptrauth (ptr @baz, i32 0)(
entry:
  call void @sink3()
  call void ptrauth (ptr @baz, i32 0)(ptr null)
  ret void
}

define void @g_addr_ADDR() {
; CHECK-LABEL: define void @g_addr_ADDR()
; CHECK:       call void @sink3()
; CHECK:       call void ptrauth (ptr @baz, i32 0, i64 0, ptr @ADDR)(
entry:
  call void @sink3()
  call void ptrauth (ptr @baz, i32 0, i64 0, ptr @ADDR)(ptr null)
  ret void
}
