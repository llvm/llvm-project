; RUN: llc < %s -mtriple=i686-pc-linux-gnu | FileCheck %s

; These static initializers are too big to hand off to assemblers
; as monolithic blobs.

; CHECK: -1
; CHECK-NEXT: -1
; CHECK: -1
; CHECK-NEXT: -1
; CHECK-NEXT: -1
; CHECK-NEXT: -1
; CHECK: -1
; CHECK-NEXT: -1
; CHECK-NEXT: -1
; CHECK-NEXT: -1
; CHECK-NEXT: -1
; CHECK-NEXT: -1
; CHECK-NEXT: -1
@x = global i128 -1
@y = global i256 -1
@z = global i512 -1
