; RUN: llc -mtriple=arm64_32-apple-ios7.0 %s -o - | FileCheck %s

define void @pass_pointer(i64 %in) {
; CHECK-LABEL: pass_pointer:
; CHECK: and x0, x0, #0xffffffff
; CHECK: bl _take_pointer

  %in32 = trunc i64 %in to i32
  %ptr = inttoptr i32 %in32 to i8*
  call i64 @take_pointer(i8* %ptr)
  ret void
}

define i64 @take_pointer(i8* %ptr) nounwind {
; CHECK-LABEL: take_pointer:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: ret

  %val = ptrtoint i8* %ptr to i32
  %res = zext i32 %val to i64
  ret i64 %res
}

define i32 @callee_ptr_stack_slot([8 x i64], i8*, i32 %val) {
; CHECK-LABEL: callee_ptr_stack_slot:
; CHECK: ldr w0, [sp, #4]

  ret i32 %val
}

define void @caller_ptr_stack_slot(i8* %ptr) {
; CHECK-LABEL: caller_ptr_stack_slot:
; CHECK-DAG: mov [[VAL:w[0-9]]], #42
; CHECK: stp w0, [[VAL]], [sp]

  call i32 @callee_ptr_stack_slot([8 x i64] undef, i8* %ptr, i32 42)
  ret void
}

define i8* @return_ptr(i64 %in, i64 %r) {
; CHECK-LABEL: return_ptr:
; CHECK: sdiv [[VAL64:x[0-9]+]], x0, x1
; CHECK: and x0, [[VAL64]], #0xffffffff

  %sum = sdiv i64 %in, %r
  %sum32 = trunc i64 %sum to i32
  %res = inttoptr i32 %sum32 to i8*
  ret i8* %res
}
