; RUN: llc < %s -mtriple=i686-windows | FileCheck %s -check-prefix=CHECK -check-prefix=NORMAL
; RUN: llc < %s -mtriple=i686-windows -mattr=call-reg-indirect | FileCheck %s -check-prefix=CHECK -check-prefix=SLM

declare void @foo(i32 %r)

define void @test(i32 %a, i32 %b) optsize nounwind {
; CHECK-LABEL: test:
; CHECK: movl [[EAX:%e..]], (%esp)
; CHECK-NEXT: pushl [[EAX]]
; CHECK-NEXT: calll
; CHECK-NEXT: addl $4, %esp
; CHECK: nop
; NORMAL: pushl (%esp)
; SLM: movl (%esp), [[RELOAD:%e..]]
; SLM-NEXT: pushl [[RELOAD]]
; CHECK: calll
; CHECK-NEXT: addl $8, %esp
  %c = add i32 %a, %b
  call void @foo(i32 %c)
  call void asm sideeffect "nop", "~{ax},~{bx},~{cx},~{dx},~{bp},~{si},~{di}"()
  call void @foo(i32 %c)
  ret void
}

define void @test_min(i32 %a, i32 %b) minsize nounwind {
; CHECK-LABEL: test_min:
; CHECK: movl [[EAX:%e..]], (%esp)
; CHECK-NEXT: pushl [[EAX]]
; CHECK-NEXT: calll
; CHECK-NEXT: popl
; CHECK: nop
; CHECK: pushl (%esp)
; CHECK: calll
; CHECK-NEXT: popl
  %c = add i32 %a, %b
  call void @foo(i32 %c)
  call void asm sideeffect "nop", "~{ax},~{bx},~{cx},~{dx},~{bp},~{si},~{di}"()
  call void @foo(i32 %c)
  ret void
}
