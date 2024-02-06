; RUN: llc -mtriple=arm64-apple-iphoneos -stop-after=finalize-isel %s -o - | FileCheck %s

declare ptr @llvm.ptrmask.p0.i64(ptr , i64)

; CHECK-LABEL: name: test1
; CHECK:         %0:gpr64 = COPY $x0
; CHECK-NEXT:    %1:gpr64sp = ANDXri %0, 8052
; CHECK-NEXT:    $x0 = COPY %1
; CHECK-NEXT:    RET_ReallyLR implicit $x0

define ptr @test1(ptr %src) {
  %ptr = call ptr @llvm.ptrmask.p0.i64(ptr %src, i64 72057594037927928)
  ret ptr %ptr
}
