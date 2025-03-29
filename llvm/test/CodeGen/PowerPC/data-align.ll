; RUN: llc < %s -mtriple=powerpc-unknown-linux | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux | FileCheck %s

; CHECK:      .set .Li8,
; CHECK-NEXT:  .size	.Li8, 1
@i8 = private constant i8 42

; CHECK:      .set .Li16,
; CHECK-NEXT: .size	.Li16, 2
@i16 = private constant i16 42

; CHECK:      .set .Li32,
; CHECK-NEXT: .size	.Li32, 4
@i32 = private constant i32 42

; CHECK:      .set .Li64,
; CHECK-NEXT: .size	.Li64, 8
@i64 = private constant i64 42

; CHECK:        .set .Li128,
; CHECK-NEXT:	.size	.Li128, 16
@i128 = private constant i128 42

