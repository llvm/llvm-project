; RUN: llc < %s -march=nvptx -mcpu=sm_35 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_35 | %ptxas-verify %}

; 64-bit divides and rems should be split into a fast and slow path where
; the fast path uses a 32-bit operation.

define void @sdiv64(i64 %a, i64 %b, ptr %retptr) {
; CHECK-LABEL: sdiv64(
; CHECK:        div.s64
; CHECK:        div.u32
; CHECK:        ret
  %d = sdiv i64 %a, %b
  store i64 %d, ptr %retptr
  ret void
}

define void @udiv64(i64 %a, i64 %b, ptr %retptr) {
; CHECK-LABEL: udiv64(
; CHECK:        div.u64
; CHECK:        div.u32
; CHECK:        ret
  %d = udiv i64 %a, %b
  store i64 %d, ptr %retptr
  ret void
}

define void @srem64(i64 %a, i64 %b, ptr %retptr) {
; CHECK-LABEL: srem64(
; CHECK:        rem.s64
; CHECK:        rem.u32
; CHECK:        ret
  %d = srem i64 %a, %b
  store i64 %d, ptr %retptr
  ret void
}

define void @urem64(i64 %a, i64 %b, ptr %retptr) {
; CHECK-LABEL: urem64(
; CHECK:        rem.u64
; CHECK:        rem.u32
; CHECK:        ret
  %d = urem i64 %a, %b
  store i64 %d, ptr %retptr
  ret void
}

define void @sdiv32(i32 %a, i32 %b, ptr %retptr) {
; CHECK-LABEL: sdiv32(
; CHECK: div.s32
; CHECK-NOT: div.
  %d = sdiv i32 %a, %b
  store i32 %d, ptr %retptr
  ret void
}

define void @udiv32(i32 %a, i32 %b, ptr %retptr) {
; CHECK-LABEL: udiv32(
; CHECK: div.u32
; CHECK-NOT: div.
  %d = udiv i32 %a, %b
  store i32 %d, ptr %retptr
  ret void
}

define void @srem32(i32 %a, i32 %b, ptr %retptr) {
; CHECK-LABEL: srem32(
; CHECK: rem.s32
; CHECK-NOT: rem.
  %d = srem i32 %a, %b
  store i32 %d, ptr %retptr
  ret void
}

define void @urem32(i32 %a, i32 %b, ptr %retptr) {
; CHECK-LABEL: urem32(
; CHECK: rem.u32
; CHECK-NOT: rem.
  %d = urem i32 %a, %b
  store i32 %d, ptr %retptr
  ret void
}
