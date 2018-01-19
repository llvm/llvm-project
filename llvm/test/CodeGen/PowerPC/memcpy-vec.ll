; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s -check-prefix=PWR7
; RUN: llc -verify-machineinstrs -mcpu=pwr8 < %s | FileCheck %s -check-prefix=PWR8
; RUN: llc -verify-machineinstrs -mcpu=a2q < %s | FileCheck %s -check-prefix=A2Q
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo1(double* nocapture %x, double* nocapture readonly %y) #0 {
entry:
  %0 = bitcast double* %x to i8*
  %1 = bitcast double* %y to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 32, i1 false)
  ret void

; PWR7-LABEL: @foo1
; PWR7-NOT: bl memcpy
; PWR7-DAG: li [[OFFSET:[0-9]+]], 16
; PWR7-DAG: lxvd2x [[TMP0:[0-9]+]], 4, [[OFFSET]]
; PWR7-DAG: stxvd2x [[TMP0]], 0, 3
; PWR7-DAG: lxvd2x [[TMP1:[0-9]+]], 0, 4
; PWR7-DAG: stxvd2x [[TMP1]], 0, 3
; PWR7: blr

; PWR8-LABEL: @foo1
; PWR8: lxvw4x
; PWR8: stxvw4x
; PWR8: blr

; A2Q-LABEL: @foo1
; A2Q-NOT: bl memcpy
; A2Q: ld {{[0-9]+}}, {{[0-9]+}}(4)
; A2Q: std {{[0-9]+}}, {{[0-9]+}}(3)
; A2Q: blr
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #0

; Function Attrs: nounwind
define void @foo2(double* nocapture %x, double* nocapture readonly %y) #0 {
entry:
  %0 = bitcast double* %x to i8*
  %1 = bitcast double* %y to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 128, i1 false)
  ret void

; PWR7-LABEL: @foo2
; PWR7: bl memcpy
; PWR7: blr

; PWR8-LABEL: @foo2
; PWR8: lxvw4x
; PWR8: stxvw4x
; PWR8: blr

; A2Q-LABEL: @foo2
; A2Q-NOT: bl memcpy
; A2Q: ld {{[0-9]+}}, {{[0-9]+}}(4)
; A2Q: std {{[0-9]+}}, {{[0-9]+}}(3)
; A2Q: blr
}

; Function Attrs: nounwind
define void @bar1(double* nocapture %x) #0 {
entry:
  %0 = bitcast double* %x to i8*
  tail call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 128, i1 false)
  ret void

; PWR7-LABEL: @bar1
; PWR7-NOT: bl memset
; PWR7: stxvw4x
; PWR7: blr

; PWR8-LABEL: @bar1
; PWR8-NOT: bl memset
; PWR8: stxvw4x
; PWR8: blr

; A2Q-LABEL: @bar1
; A2Q-NOT: bl memset
; A2Q: std {{[0-9]+}}, {{[0-9]+}}(3)
; A2Q: blr
}

; Function Attrs: nounwind
define void @bar2(double* nocapture %x) #0 {
entry:
  %0 = bitcast double* %x to i8*
  tail call void @llvm.memset.p0i8.i64(i8* align 32 %0, i8 0, i64 128, i1 false)
  ret void

; PWR7-LABEL: @bar2
; PWR7-NOT: bl memset
; PWR7: stxvw4x
; PWR7: blr

; PWR8-LABEL: @bar2
; PWR8-NOT: bl memset
; PWR8: stxvw4x
; PWR8: blr

; A2Q-LABEL: @bar2
; A2Q-NOT: bl memset
; A2Q: qvstfdx
; A2Q: blr
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) #0

attributes #0 = { nounwind }

