; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le"

; Function Attrs: nounwind
define void @test_vsx() unnamed_addr #0 align 2 {
entry:
  %0 = load i32, i32* undef, align 4
  %1 = trunc i32 %0 to i8
  call void @llvm.memset.p0i8.i64(i8* null, i8 %1, i64 32, i1 false)
  ret void

; CHECK-LABEL: @test_vsx
; CHECK: stxvd2x
; CHECK: stxvd2x
; CHECK: blr
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) #1

attributes #0 = { nounwind "target-cpu"="pwr8" }
attributes #1 = { nounwind }

