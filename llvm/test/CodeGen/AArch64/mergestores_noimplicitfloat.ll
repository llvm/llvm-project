; RUN: llc -o - %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios10.0.0"

; PR33475 - Expect 64-bit operations as 128-operations are not legal

; CHECK-LABEL: pr33475
; CHECK-DAG: ldr [[R0:x[0-9]+]], [x1]
; CHECK-DAG: str [[R0]], [x0]
; CHECK-DAG: ldr [[R1:x[0-9]+]], [x1, #8]
; CHECK-DAG: str [[R1]], [x0, #8]
; CHECK-DAG: ldr [[R2:x[0-9]+]], [x1, #16]
; CHECK-DAG: str [[R2]], [x0, #16]
; CHECK-DAG: ldr [[R3:x[0-9]+]], [x1, #24]
; CHECK-DAG: str [[R3]], [x0, #24]

define void @pr33475(i8* %p0, i8* %p1) noimplicitfloat {
    call void @llvm.memcpy.p0i8.p0i8.i64(i8* %p0, i8* %p1, i64 32, i32 4, i1 false)
    ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)
