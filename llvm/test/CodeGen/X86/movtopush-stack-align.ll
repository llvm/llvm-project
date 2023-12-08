; RUN: llc < %s -mtriple=i686-windows -stackrealign | FileCheck %s

declare void @good(i32 %a, i32 %b, i32 %c, i32 %d)
declare void @oneparam(i32 %a)
declare void @eightparams(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h)

; When there is no reserved call frame, check that additional alignment
; is added when the pushes don't add up to the required alignment.
; CHECK-LABEL: test5:
; CHECK: subl    $16, %esp
; CHECK-NEXT: pushl   $4
; CHECK-NEXT: pushl   $3
; CHECK-NEXT: pushl   $2
; CHECK-NEXT: pushl   $1
; CHECK-NEXT: call
define void @test5(i32 %k) {
entry:
  %a = alloca i32, i32 %k
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; When the alignment adds up, do the transformation
; CHECK-LABEL: test5b:
; CHECK: pushl   $8
; CHECK-NEXT: pushl   $7
; CHECK-NEXT: pushl   $6
; CHECK-NEXT: pushl   $5
; CHECK-NEXT: pushl   $4
; CHECK-NEXT: pushl   $3
; CHECK-NEXT: pushl   $2
; CHECK-NEXT: pushl   $1
; CHECK-NEXT: call
define void @test5b() optsize {
entry:
  call void @eightparams(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8)
  ret void
}

; When having to compensate for the alignment isn't worth it,
; don't use pushes.
; CHECK-LABEL: test5c:
; CHECK: movl $1, (%esp)
; CHECK-NEXT: call
define void @test5c() optsize {
entry:
  call void @oneparam(i32 1)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"override-stack-alignment", i32 32}
