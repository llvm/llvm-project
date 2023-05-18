; RUN: llc -mtriple=aarch64-unknown-unknown                             -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-unknown-unknown -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

define void @store_i8(ptr %a) {
; CHECK-LABEL: store_i8
; CHECK: strb  wzr, [x0]
  store i8 0, ptr %a
  ret void
}

define void @store_i16(ptr %a) {
; CHECK-LABEL: store_i16
; CHECK: strh  wzr, [x0]
  store i16 0, ptr %a
  ret void
}

define void @store_i32(ptr %a) {
; CHECK-LABEL: store_i32
; CHECK: str  wzr, [x0]
  store i32 0, ptr %a
  ret void
}

define void @store_i64(ptr %a) {
; CHECK-LABEL: store_i64
; CHECK: str  xzr, [x0]
  store i64 0, ptr %a
  ret void
}
