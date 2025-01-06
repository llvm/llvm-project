; RUN: llc < %s -mtriple=sparc | FileCheck %s
; RUN: llc < %s -mtriple=sparcel | FileCheck %s
; RUN: llc < %s -mtriple=sparcv9 | FileCheck %s

; CHECK:      .Li8:
; CHECK-DAG: .size .Li8, 1
@i8 = private constant i8 42

; CHECK:      .p2align 1
; CHECK-NEXT: .Li16:
; CHECK-DAG:  .size .Li16, 2
@i16 = private constant i16 42

; CHECK:      .p2align 2
; CHECK-NEXT: .Li32:
; CHECK-DAG:  .size .Li32, 4
@i32 = private constant i32 42

; CHECK:      .p2align 3
; CHECK-NEXT: .Li64:
; CHECK-DAG:  .size .Li64, 8
@i64 = private constant i64 42

; CHECK:      .p2align 4
; CHECK-NEXT: .Li128:
; CHECK-DAG:  .size .Li128, 16
@i128 = private constant i128 42
