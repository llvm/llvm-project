; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s --check-prefixes CHECK,SLOW
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -fast-isel -fast-isel-abort=1 | FileCheck %s --check-prefixes CHECK,FAST

; Test that basic 64-bit integer comparison operations assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: eq_i64:
; CHECK-NEXT: .functype eq_i64 (i64, i64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.eq $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SLOW-NEXT:  return $pop[[L2]]{{$}}
; FAST-NEXT:  i32.const $push[[L3:[0-9]+]]=, 1{{$}}
; FAST-NEXT:  i32.and $push[[L4:[0-9]+]]=, $pop[[L2]], $pop[[L3]]{{$}}
; FAST-NEXT:  return $pop[[L4]]{{$}}
define i32 @eq_i64(i64 %x, i64 %y) {
  %a = icmp eq i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ne_i64:
; CHECK: i64.ne $push[[L0:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; SLOW-NEXT: return $pop[[L0]]{{$}}
; FAST-NEXT: i32.const $push[[L1:[0-9]+]]=, 1{{$}}
; FAST-NEXT: i32.and $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; FAST-NEXT: return $pop[[L2]]{{$}}
define i32 @ne_i64(i64 %x, i64 %y) {
  %a = icmp ne i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: slt_i64:
; CHECK: i64.lt_s $push[[L0:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; SLOW-NEXT: return $pop[[L0]]{{$}}
; FAST-NEXT: i32.const $push[[L1:[0-9]+]]=, 1{{$}}
; FAST-NEXT: i32.and $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; FAST-NEXT: return $pop[[L2]]{{$}}
define i32 @slt_i64(i64 %x, i64 %y) {
  %a = icmp slt i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sle_i64:
; CHECK: i64.le_s $push[[L0:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; SLOW-NEXT: return $pop[[L0]]{{$}}
; FAST-NEXT: i32.const $push[[L1:[0-9]+]]=, 1{{$}}
; FAST-NEXT: i32.and $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; FAST-NEXT: return $pop[[L2]]{{$}}
define i32 @sle_i64(i64 %x, i64 %y) {
  %a = icmp sle i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_i64:
; CHECK: i64.lt_u $push[[L0:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; SLOW-NEXT: return $pop[[L0]]{{$}}
; FAST-NEXT: i32.const $push[[L1:[0-9]+]]=, 1{{$}}
; FAST-NEXT: i32.and $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; FAST-NEXT: return $pop[[L2]]{{$}}
define i32 @ult_i64(i64 %x, i64 %y) {
  %a = icmp ult i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_i64:
; CHECK: i64.le_u $push[[L0:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; SLOW-NEXT: return $pop[[L0]]{{$}}
; FAST-NEXT: i32.const $push[[L1:[0-9]+]]=, 1{{$}}
; FAST-NEXT: i32.and $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; FAST-NEXT: return $pop[[L2]]{{$}}
define i32 @ule_i64(i64 %x, i64 %y) {
  %a = icmp ule i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sgt_i64:
; CHECK: i64.gt_s $push[[L0:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; SLOW-NEXT: return $pop[[L0]]{{$}}
; FAST-NEXT: i32.const $push[[L1:[0-9]+]]=, 1{{$}}
; FAST-NEXT: i32.and $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; FAST-NEXT: return $pop[[L2]]{{$}}
define i32 @sgt_i64(i64 %x, i64 %y) {
  %a = icmp sgt i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sge_i64:
; CHECK: i64.ge_s $push[[L0:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; SLOW-NEXT: return $pop[[L0]]{{$}}
; FAST-NEXT: i32.const $push[[L1:[0-9]+]]=, 1{{$}}
; FAST-NEXT: i32.and $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; FAST-NEXT: return $pop[[L2]]{{$}}
define i32 @sge_i64(i64 %x, i64 %y) {
  %a = icmp sge i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_i64:
; CHECK: i64.gt_u $push[[L0:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; SLOW-NEXT: return $pop[[L0]]{{$}}
; FAST-NEXT: i32.const $push[[L1:[0-9]+]]=, 1{{$}}
; FAST-NEXT: i32.and $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; FAST-NEXT: return $pop[[L2]]{{$}}
define i32 @ugt_i64(i64 %x, i64 %y) {
  %a = icmp ugt i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_i64:
; CHECK: i64.ge_u $push[[L0:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; SLOW-NEXT: return $pop[[L0]]{{$}}
; FAST-NEXT: i32.const $push[[L1:[0-9]+]]=, 1{{$}}
; FAST-NEXT: i32.and $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; FAST-NEXT: return $pop[[L2]]{{$}}
define i32 @uge_i64(i64 %x, i64 %y) {
  %a = icmp uge i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
