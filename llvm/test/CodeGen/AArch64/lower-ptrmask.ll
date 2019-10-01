; RUN: llc -mtriple=arm64-apple-iphoneos -stop-after=finalize-isel %s -o - | FileCheck %s

declare i8* @llvm.ptrmask.p0i8.i64(i8* , i64)

; CHECK-LABEL: name: test1
; CHECK:         %0:gpr64 = COPY $x0
; CHECK-NEXT:    %1:gpr64sp = ANDXri %0, 8052
; CHECK-NEXT:    $x0 = COPY %1
; CHECK-NEXT:    RET_ReallyLR implicit $x0

define i8* @test1(i8* %src) {
  %ptr = call i8* @llvm.ptrmask.p0i8.i64(i8* %src, i64 72057594037927928)
  ret i8* %ptr
}

declare i8* @llvm.ptrmask.p0i8.i32(i8*, i32)

; CHECK-LABEL: name: test2
; CHECK:         %0:gpr64 = COPY $x0
; CHECK-NEXT:    %1:gpr32 = MOVi32imm 10000
; CHECK-NEXT:    %2:gpr64 = SUBREG_TO_REG 0, killed %1, %subreg.sub_32
; CHECK-NEXT:    %3:gpr64 = ANDXrr %0, killed %2
; CHECK-NEXT:    $x0 = COPY %3
; CHECK-NEXT:    RET_ReallyLR implicit $x0

define i8* @test2(i8* %src) {
  %ptr = call i8* @llvm.ptrmask.p0i8.i32(i8* %src, i32 10000)
  ret i8* %ptr
}
