; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

@a = internal global i32 0, align 4
@b = internal global i64 0, align 8
@c = internal global i16 0, align 2

; CHECK:      .lcomm a,4,a,2
; CHECK-NEXT: .lcomm b,8,b,3
; CHECK-NEXT: .lcomm c,2,c,1
