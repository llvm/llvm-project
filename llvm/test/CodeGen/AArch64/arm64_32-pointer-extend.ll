; RUN: llc -mtriple=arm64_32-apple-ios7.0 %s -o - | FileCheck %s

define void @pass_pointer(i64 %in) {
; CHECK-LABEL: pass_pointer:
; CHECK: mov w0, w0
; CHECK: bl _take_pointer

  %in32 = trunc i64 %in to i32
  %ptr = inttoptr i32 %in32 to ptr
  call i64 @take_pointer(ptr %ptr)
  ret void
}

define i64 @take_pointer(ptr %ptr) nounwind {
; CHECK-LABEL: take_pointer:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: ret

  %val = ptrtoint ptr %ptr to i32
  %res = zext i32 %val to i64
  ret i64 %res
}

define i32 @callee_ptr_stack_slot([8 x i64], ptr, i32 %val) {
; CHECK-LABEL: callee_ptr_stack_slot:
; CHECK: ldr w0, [sp, #4]

  ret i32 %val
}

define void @caller_ptr_stack_slot(ptr %ptr) {
; CHECK-LABEL: caller_ptr_stack_slot:
; CHECK-DAG: mov [[VAL:w[0-9]]], #42
; CHECK: stp w0, [[VAL]], [sp]

  call i32 @callee_ptr_stack_slot([8 x i64] undef, ptr %ptr, i32 42)
  ret void
}

define ptr @return_ptr(i64 %in, i64 %r) {
; CHECK-LABEL: return_ptr:
; CHECK: sdiv x[[VAL64:[0-9]+]], x0, x1
; CHECK: mov w0, w[[VAL64]]

  %sum = sdiv i64 %in, %r
  %sum32 = trunc i64 %sum to i32
  %res = inttoptr i32 %sum32 to ptr
  ret ptr %res
}
