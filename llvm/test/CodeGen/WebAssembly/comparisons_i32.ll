; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt | FileCheck %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -fast-isel -fast-isel-abort=1 | FileCheck %s

; Test that basic 32-bit integer comparison operations assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; CHECK-LABEL: eq_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: get_local $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: get_local $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.eq $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @eq_i32(i32 %x, i32 %y) {
  %a = icmp eq i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ne_i32:
; CHECK: i32.ne $push[[NUM:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ne_i32(i32 %x, i32 %y) {
  %a = icmp ne i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: slt_i32:
; CHECK: i32.lt_s $push[[NUM:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @slt_i32(i32 %x, i32 %y) {
  %a = icmp slt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sle_i32:
; CHECK: i32.le_s $push[[NUM:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @sle_i32(i32 %x, i32 %y) {
  %a = icmp sle i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_i32:
; CHECK: i32.lt_u $push[[NUM:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ult_i32(i32 %x, i32 %y) {
  %a = icmp ult i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_i32:
; CHECK: i32.le_u $push[[NUM:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ule_i32(i32 %x, i32 %y) {
  %a = icmp ule i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sgt_i32:
; CHECK: i32.gt_s $push[[NUM:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @sgt_i32(i32 %x, i32 %y) {
  %a = icmp sgt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sge_i32:
; CHECK: i32.ge_s $push[[NUM:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @sge_i32(i32 %x, i32 %y) {
  %a = icmp sge i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_i32:
; CHECK: i32.gt_u $push[[NUM:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ugt_i32(i32 %x, i32 %y) {
  %a = icmp ugt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_i32:
; CHECK: i32.ge_u $push[[NUM:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @uge_i32(i32 %x, i32 %y) {
  %a = icmp uge i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
