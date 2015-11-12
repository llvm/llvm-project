; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 32-bit integer comparison operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: eq_i32:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32, i32, i32{{$}}
; CHECK-NEXT: i32.eq $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i32 @eq_i32(i32 %x, i32 %y) {
  %a = icmp eq i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ne_i32:
; CHECK: i32.ne $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
define i32 @ne_i32(i32 %x, i32 %y) {
  %a = icmp ne i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: slt_i32:
; CHECK: i32.lt_s $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
define i32 @slt_i32(i32 %x, i32 %y) {
  %a = icmp slt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sle_i32:
; CHECK: i32.le_s $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
define i32 @sle_i32(i32 %x, i32 %y) {
  %a = icmp sle i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_i32:
; CHECK: i32.lt_u $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
define i32 @ult_i32(i32 %x, i32 %y) {
  %a = icmp ult i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_i32:
; CHECK: i32.le_u $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
define i32 @ule_i32(i32 %x, i32 %y) {
  %a = icmp ule i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sgt_i32:
; CHECK: i32.gt_s $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
define i32 @sgt_i32(i32 %x, i32 %y) {
  %a = icmp sgt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sge_i32:
; CHECK: i32.ge_s $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
define i32 @sge_i32(i32 %x, i32 %y) {
  %a = icmp sge i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_i32:
; CHECK: i32.gt_u $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
define i32 @ugt_i32(i32 %x, i32 %y) {
  %a = icmp ugt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_i32:
; CHECK: i32.ge_u $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
define i32 @uge_i32(i32 %x, i32 %y) {
  %a = icmp uge i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
