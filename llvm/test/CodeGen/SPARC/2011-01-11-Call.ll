; RUN: llc -march=sparc -O0 <%s
; RUN: llc -march=sparc   <%s | FileCheck %s --check-prefix=V8
; RUN: llc -march=sparcv9 <%s | FileCheck %s --check-prefix=V9

; V8-LABEL: test
; V8:       save %sp
; V8:       call foo
; V8-NEXT:  nop
; V8:       call bar
; V8-NEXT:  nop
; V8:       ret
; V8-NEXT:  restore

; V9-LABEL: test
; V9:       save %sp
; V9:       call foo
; V9-NEXT:  nop
; V9:       call bar
; V9-NEXT:  nop
; V9:       ret
; V9-NEXT:  restore

define void @test() #0 {
entry:
 %0 = tail call i32 (...) @foo() nounwind
 tail call void (...) @bar() nounwind
 ret void
}

declare i32 @foo(...)

declare void @bar(...)

; V8-LABEL: test_tail_call_with_return
; V8:       mov %o7, %g1
; V8-NEXT:  call foo
; V8-NEXT:  mov %g1, %o7

; V9-LABEL: test_tail_call_with_return
; V9:       save %sp
; V9:       call foo
; V9-NEXT:  nop
; V9:       ret
; V9-NEXT:  restore %g0, %o0, %o0

define i32 @test_tail_call_with_return() nounwind {
entry:
 %0 = tail call i32 (...) @foo() nounwind
 ret i32 %0
}

attributes #0 = { nounwind "disable-tail-calls"="true" }
