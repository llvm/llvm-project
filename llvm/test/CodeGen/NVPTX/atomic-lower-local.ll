; RUN: opt < %s -S -nvptx-atomic-lower | FileCheck %s

; This test ensures that there is a legal way for ptx to lower atomics
; on local memory. Here, we demonstrate this by lowering them to simple
; load and stores.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

define { i32, i1 } @test_cmpxchg(ptr addrspace(5) %ptr, i32 %cmp, i32 %val) {
; CHECK-LABEL: @test_cmpxchg(
; CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr addrspace(5) [[PTR:%.*]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], [[CMP:%.*]]
; CHECK-NEXT:    [[TMP3:%.*]] = select i1 [[TMP2]], i32 [[VAL:%.*]], i32 [[TMP1]]
; CHECK-NEXT:    store i32 [[TMP3]], ptr addrspace(5) [[PTR]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = insertvalue { i32, i1 } poison, i32 [[TMP1]], 0
; CHECK-NEXT:    [[TMP5:%.*]] = insertvalue { i32, i1 } [[TMP4]], i1 [[TMP2]], 1
; CHECK-NEXT:    ret { i32, i1 } [[TMP5]]
;
  %res = cmpxchg ptr addrspace(5) %ptr, i32 %cmp, i32 %val monotonic monotonic
  ret { i32, i1 } %res
}

define double @test_atomicrmw(ptr addrspace(5) %ptr, double %val) {
; CHECK-LABEL: @test_atomicrmw(
; CHECK-NEXT:    [[TMP1:%.*]] = load double, ptr addrspace(5) [[PTR:%.*]], align 8
; CHECK-NEXT:    [[NEW:%.*]] = fadd double [[TMP1]], [[VAL:%.*]]
; CHECK-NEXT:    store double [[NEW]], ptr addrspace(5) [[PTR]], align 8
; CHECK-NEXT:    ret double [[TMP1]]
;
  %res = atomicrmw fadd ptr addrspace(5) %ptr, double %val monotonic, align 8
  ret double %res
}
