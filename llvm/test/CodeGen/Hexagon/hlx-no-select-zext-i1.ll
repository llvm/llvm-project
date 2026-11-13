; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -verify-machineinstrs \
; RUN:   < %s | FileCheck %s

; Check that an operation on an HLX vector with a zext of an i1 operand is not
; turned into a select of two operations. The select would be created after
; legalization and could not be selected.

; CHECK-LABEL: zext_i1_shift:
; CHECK: L{{[0-9]+}}.h = sub(
; CHECK: L{{[0-9]+}}.h = shr(
define void @zext_i1_shift(ptr %pa, ptr %out, i32 %x, i32 %y) {
  %a = load <1024 x i16>, ptr %pa, align 2048
  %cmp = icmp sgt i32 %x, %y
  %z = zext i1 %cmp to i16
  %splat.insert = insertelement <1024 x i16> poison, i16 %z, i64 0
  %splat = shufflevector <1024 x i16> %splat.insert, <1024 x i16> poison, <1024 x i32> zeroinitializer
  %result = shl <1024 x i16> %a, %splat
  store <1024 x i16> %result, ptr %out, align 2048
  ret void
}
