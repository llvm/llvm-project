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

declare ptr @llvm.ptrmask.p0.i32(ptr, i32)

; CHECK-LABEL: name: test2
; CHECK:         %0:gpr64 = COPY $x0
; CHECK-NEXT:    %1:gpr32 = MOVi32imm 10000
; CHECK-NEXT:    %2:gpr64 = SUBREG_TO_REG 0, killed %1, %subreg.sub_32
; CHECK-NEXT:    %3:gpr64 = ANDXrr %0, killed %2
; CHECK-NEXT:    $x0 = COPY %3
; CHECK-NEXT:    RET_ReallyLR implicit $x0

define ptr @test2(ptr %src) {
  %ptr = call ptr @llvm.ptrmask.p0.i32(ptr %src, i32 10000)
  ret ptr %ptr
}
