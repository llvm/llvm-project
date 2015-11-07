; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit integer comparison operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: eq_i64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i64, i64, i32{{$}}
; CHECK-NEXT: get_local push, 1{{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 3, pop{{$}}
; CHECK-NEXT: i64.eq push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
; CHECK-NEXT: return (get_local 4){{$}}
define i32 @eq_i64(i64 %x, i64 %y) {
  %a = icmp eq i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ne_i64:
; CHECK: i64.ne push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @ne_i64(i64 %x, i64 %y) {
  %a = icmp ne i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: slt_i64:
; CHECK: i64.lt_s push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @slt_i64(i64 %x, i64 %y) {
  %a = icmp slt i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sle_i64:
; CHECK: i64.le_s push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @sle_i64(i64 %x, i64 %y) {
  %a = icmp sle i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_i64:
; CHECK: i64.lt_u push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @ult_i64(i64 %x, i64 %y) {
  %a = icmp ult i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_i64:
; CHECK: i64.le_u push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @ule_i64(i64 %x, i64 %y) {
  %a = icmp ule i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sgt_i64:
; CHECK: i64.gt_s push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @sgt_i64(i64 %x, i64 %y) {
  %a = icmp sgt i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sge_i64:
; CHECK: i64.ge_s push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @sge_i64(i64 %x, i64 %y) {
  %a = icmp sge i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_i64:
; CHECK: i64.gt_u push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @ugt_i64(i64 %x, i64 %y) {
  %a = icmp ugt i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_i64:
; CHECK: i64.ge_u push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @uge_i64(i64 %x, i64 %y) {
  %a = icmp uge i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
