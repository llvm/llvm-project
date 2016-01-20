; RUN: llc -mtriple=x86_64-apple-macosx -O3 -enable-implicit-null-checks -o - < %s 2>&1 | FileCheck %s

declare void @throw0()
declare void @throw1()

define i1 @f(i8* %p0, i8* %p1) {
 entry:
  %c0 = icmp eq i8* %p0, null
  br i1 %c0, label %throw0, label %continue0, !make.implicit !0

 continue0:
  %v0 = load i8, i8* %p0
  %c1 = icmp eq i8* %p1, null
  br i1 %c1, label %throw1, label %continue1, !make.implicit !0

 continue1:
  %v1 = load i8, i8* %p1
  %v = icmp eq i8 %v0, %v1
  ret i1 %v

 throw0:
  call void @throw0()
  unreachable

 throw1:
  call void @throw1()
  unreachable
}

; Check that we have two implicit null checks in @f

; CHECK: __LLVM_FaultMaps:
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .short  0
; CHECK-NEXT:        .long   1

; FunctionInfo[0] =

; FunctionAddress =
; CHECK-NEXT:        .quad   _f

; NumFaultingPCs =
; CHECK-NEXT:        .long   2

; Reserved =
; CHECK-NEXT:        .long   0

!0 = !{}
