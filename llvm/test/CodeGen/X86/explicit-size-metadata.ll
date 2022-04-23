; RUN: llc < %s -mtriple=x86_64 | FileCheck %s

@a = global i64 0, align 8, !explicit_size !0
; CHECK: .size a, 4

@b = global i64 0, align 8
; CHECK: .size b, 8

@larger = global i16 0, align 4, !explicit_size !0
; CHECK: .size larger, 4

@array = global { [8 x i8] } zeroinitializer, !explicit_size !0
; CHECK: .size array, 4

!0 = !{i64 4}
