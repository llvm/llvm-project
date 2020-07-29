; RUN: llc -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 < %s | FileCheck %s -check-prefix=CHECK-O0
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-bgq-linux"

; Function Attrs: nounwind
define void @test_qpx() unnamed_addr #0 align 2 {
entry:
  %0 = load i32, i32* undef, align 4
  %1 = trunc i32 %0 to i8
  call void @llvm.memset.p0i8.i64(i8* align 32 null, i8 %1, i64 64, i1 false)
  ret void

; CHECK-LABEL: @test_qpx
; CHECK: qvstfdx
; CHECK: qvstfdx
; CHECK: blr

; CHECK-O0-LABEL: @test_qpx
; CHECK-O0-NOT: qvstfdx
; CHECK-O0: blr
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) #1

; Function Attrs: nounwind
define void @test_vsx() unnamed_addr #2 align 2 {
entry:
  %0 = load i32, i32* undef, align 4
  %1 = trunc i32 %0 to i8
  call void @llvm.memset.p0i8.i64(i8* null, i8 %1, i64 32, i1 false)
  ret void

; CHECK-LABEL: @test_vsx
; CHECK: stxvw4x
; CHECK: stxvw4x
; CHECK: blr

; CHECK-O0-LABEL: @test_vsx
; CHECK-O0-NOT: stxvw4x
; CHECK-O0: blr
}

attributes #0 = { nounwind "target-cpu"="a2q" }
attributes #1 = { nounwind }
attributes #2 = { nounwind "target-cpu"="pwr7" }

