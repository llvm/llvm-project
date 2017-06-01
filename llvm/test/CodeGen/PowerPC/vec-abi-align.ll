; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=+vsx < %s | FileCheck -check-prefix=CHECK-VSX %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.s2 = type { i64, <4 x float> }

@ve = external global <4 x float>
@n = external global i64

; Function Attrs: nounwind
define void @test1(i64 %d1, i64 %d2, i64 %d3, i64 %d4, i64 %d5, i64 %d6, i64 %d7, i64 %d8, i64 %d9, <4 x float> inreg %vs.coerce) #0 {
entry:
  store <4 x float> %vs.coerce, <4 x float>* @ve, align 16
  ret void

; CHECK-LABEL: @test1
; CHECK: stvx 2,
; CHECK: blr

; CHECK-VSX-LABEL: @test1
; CHECK-VSX: stxvw4x 34,
; CHECK-VSX: blr
}

; Function Attrs: nounwind
define void @test2(i64 %d1, i64 %d2, i64 %d3, i64 %d4, i64 %d5, i64 %d6, i64 %d7, i64 %d8, %struct.s2* byval nocapture readonly %vs) #0 {
entry:
  %m = getelementptr inbounds %struct.s2, %struct.s2* %vs, i64 0, i32 0
  %0 = load i64, i64* %m, align 8
  store i64 %0, i64* @n, align 8
  %v = getelementptr inbounds %struct.s2, %struct.s2* %vs, i64 0, i32 1
  %1 = load <4 x float>, <4 x float>* %v, align 16
  store <4 x float> %1, <4 x float>* @ve, align 16
  ret void

; CHECK-LABEL: @test2
; CHECK-DAG: ld {{[0-9]+}}, 112(1)
; CHECK-DAG: li [[REG16:[0-9]+]], 16
; CHECK-DAG: addi [[REGB:[0-9]+]], 1, 112
; CHECK-DAG: lvx 2, [[REGB]], [[REG16]]
; CHECK: blr

; CHECK-VSX-LABEL: @test2
; CHECK-VSX-DAG: ld {{[0-9]+}}, 112(1)
; CHECK-VSX-DAG: li [[REG16:[0-9]+]], 16
; CHECK-VSX-DAG: addi [[REGB:[0-9]+]], 1, 112
; CHECK-VSX-DAG: lxvw4x {{[0-9]+}}, [[REGB]], [[REG16]]
; CHECK-VSX: blr
}

; Function Attrs: nounwind
define void @test3(i64 %d1, i64 %d2, i64 %d3, i64 %d4, i64 %d5, i64 %d6, i64 %d7, i64 %d8, i64 %d9, %struct.s2* byval nocapture readonly %vs) #0 {
entry:
  %m = getelementptr inbounds %struct.s2, %struct.s2* %vs, i64 0, i32 0
  %0 = load i64, i64* %m, align 8
  store i64 %0, i64* @n, align 8
  %v = getelementptr inbounds %struct.s2, %struct.s2* %vs, i64 0, i32 1
  %1 = load <4 x float>, <4 x float>* %v, align 16
  store <4 x float> %1, <4 x float>* @ve, align 16
  ret void

; CHECK-LABEL: @test3
; CHECK-DAG: ld {{[0-9]+}}, 128(1)
; CHECK-DAG: li [[REG16:[0-9]+]], 16
; CHECK-DAG: addi [[REGB:[0-9]+]], 1, 128
; CHECK-DAG: lvx 2, [[REGB]], [[REG16]]
; CHECK: blr

; CHECK-VSX-LABEL: @test3
; CHECK-VSX-DAG: ld {{[0-9]+}}, 128(1)
; CHECK-VSX-DAG: li [[REG16:[0-9]+]], 16
; CHECK-VSX-DAG: addi [[REGB:[0-9]+]], 1, 128
; CHECK-VSX-DAG: lxvw4x {{[0-9]+}}, [[REGB]], [[REG16]]
; CHECK-VSX: blr
}

attributes #0 = { nounwind }

