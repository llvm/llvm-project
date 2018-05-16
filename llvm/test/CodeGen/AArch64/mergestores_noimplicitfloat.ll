; RUN: llc -o - %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios10.0.0"

; PR33475 - Expect 64-bit operations as 128-operations are not legal
; However, we can generate a paired 64-bit loads and stores, without using
; floating point registers.

; CHECK-LABEL: pr33475
; CHECK-DAG: ldp [[R0:x[0-9]+]], [[R0:x[0-9]+]], [x1, #16]
; CHECK-DAG: ldp [[R0:x[0-9]+]], [[R0:x[0-9]+]], [x1]
; CHECK-DAG: stp [[R0:x[0-9]+]], [[R0:x[0-9]+]], [x0, #16]
; CHECK-DAG: stp [[R0:x[0-9]+]], [[R0:x[0-9]+]], [x0]

define void @pr33475(i8* %p0, i8* %p1) noimplicitfloat {
    call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %p0, i8* align 4 %p1, i64 32, i1 false)
    ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)
