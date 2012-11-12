; RUN: opt < %s -instsimplify -S | FileCheck %s

; PR12189
define i1 @test1(i32 %x) {
; CHECK: @test1
  br i1 true, label %a, label %b

a:
  %aa = or i32 %x, 10
  br label %c

b:
  %bb = or i32 %x, 10
  br label %c

c:
  %cc = phi i32 [ %bb, %b ], [%aa, %a ]
  %d = urem i32 %cc, 2
  %e = icmp eq i32 %d, 0
  ret i1 %e
; CHECK: ret i1 %e
}
