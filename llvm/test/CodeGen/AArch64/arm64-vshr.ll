; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -mcpu=cyclone | FileCheck %s

define <8 x i16> @testShiftRightArith_v8i16(<8 x i16> %a, <8 x i16> %b) #0 {
; CHECK-LABEL: testShiftRightArith_v8i16:
; CHECK: neg.8h	[[REG1:v[0-9]+]], [[REG1]]
; CHECK-NEXT: sshl.8h [[REG2:v[0-9]+]], [[REG2]], [[REG1]]

entry:
  %a.addr = alloca <8 x i16>, align 16
  %b.addr = alloca <8 x i16>, align 16
  store <8 x i16> %a, <8 x i16>* %a.addr, align 16
  store <8 x i16> %b, <8 x i16>* %b.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %a.addr, align 16
  %1 = load <8 x i16>, <8 x i16>* %b.addr, align 16
  %shr = ashr <8 x i16> %0, %1
  ret <8 x i16> %shr
}

define <4 x i32> @testShiftRightArith_v4i32(<4 x i32> %a, <4 x i32> %b) #0 {
; CHECK-LABEL: testShiftRightArith_v4i32:
; CHECK: neg.4s	[[REG3:v[0-9]+]], [[REG3]]
; CHECK-NEXT: sshl.4s [[REG4:v[0-9]+]], [[REG4]], [[REG3]]
entry:
  %a.addr = alloca <4 x i32>, align 32
  %b.addr = alloca <4 x i32>, align 32
  store <4 x i32> %a, <4 x i32>* %a.addr, align 32
  store <4 x i32> %b, <4 x i32>* %b.addr, align 32
  %0 = load <4 x i32>, <4 x i32>* %a.addr, align 32
  %1 = load <4 x i32>, <4 x i32>* %b.addr, align 32
  %shr = ashr <4 x i32> %0, %1
  ret <4 x i32> %shr
}

define <8 x i16> @testShiftRightLogical(<8 x i16> %a, <8 x i16> %b) #0 {
; CHECK: testShiftRightLogical
; CHECK: neg.8h	[[REG5:v[0-9]+]], [[REG5]]
; CHECK-NEXT: ushl.8h [[REG6:v[0-9]+]], [[REG6]], [[REG5]]
entry:
  %a.addr = alloca <8 x i16>, align 16
  %b.addr = alloca <8 x i16>, align 16
  store <8 x i16> %a, <8 x i16>* %a.addr, align 16
  store <8 x i16> %b, <8 x i16>* %b.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %a.addr, align 16
  %1 = load <8 x i16>, <8 x i16>* %b.addr, align 16
  %shr = lshr <8 x i16> %0, %1
  ret <8 x i16> %shr
}

define <1 x i64> @sshr_v1i64(<1 x i64> %A) nounwind {
; CHECK-LABEL: sshr_v1i64:
; CHECK: sshr d0, d0, #42
  %tmp3 = ashr <1 x i64> %A, < i64 42 >
  ret <1 x i64> %tmp3
}

define <1 x i64> @cmlt_v1i64(<1 x i64> %A) nounwind {
; CHECK-LABEL: cmlt_v1i64:
; CHECK: cmlt d0, d0, #0
  %tmp3 = ashr <1 x i64> %A, < i64 63 >
  ret <1 x i64> %tmp3
}

define <1 x i64> @ushr_v1i64(<1 x i64> %A) nounwind {
; CHECK-LABEL: ushr_v1i64:
; CHECK: ushr d0, d0, #63
  %tmp3 = lshr <1 x i64> %A, < i64 63 >
  ret <1 x i64> %tmp3
}

attributes #0 = { nounwind }
