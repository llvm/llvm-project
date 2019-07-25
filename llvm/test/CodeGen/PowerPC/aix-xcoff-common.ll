; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s

@a = common global i32 0, align 4
@b = common global i64 0, align 8
@c = common global i16 0, align 2

@d = common local_unnamed_addr global double 0.000000e+00, align 8
@f = common local_unnamed_addr global float 0.000000e+00, align 4
@comm = common local_unnamed_addr global double 0.000000e+00, align 8

@over_aligned = common local_unnamed_addr global double 0.000000e+00, align 32

@array = common local_unnamed_addr global [32 x i8] zeroinitializer, align 1

; CHECK:      .csect .text[PR]
; CHECK-NEXT:  .file
; CHECK-NEXT: .comm   a,4,2
; CHECK-NEXT: .comm   b,8,3
; CHECK-NEXT: .comm   c,2,1
; CHECK-NEXT: .comm   d,8,3
; CHECK-NEXT: .comm   f,4,2
; CHECK-NEXT: .comm   comm,8,3
; CHECK-NEXT: .comm   over_aligned,8,5
; CHECK-NEXT: .comm   array,32,0
