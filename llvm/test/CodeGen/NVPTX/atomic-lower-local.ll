; RUN: opt < %s -S -nvptx-atomic-lower | FileCheck %s

; This test ensures that there is a legal way for ptx to lower atomics
; on local memory. Here, we demonstrate this by lowering them to simple
; load and stores.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

define double @kernel(double addrspace(5)* %ptr, double %val) {
  %res = atomicrmw fadd double addrspace(5)* %ptr, double %val monotonic, align 8
  ret double %res
; CHECK:   %1 = load double, double addrspace(5)* %ptr, align 8
; CHECK-NEXT:   %2 = fadd double %1, %val
; CHECK-NEXT:   store double %2, double addrspace(5)* %ptr, align 8
; CHECK-NEXT:   ret double %1
}

