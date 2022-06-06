; Test that incorrect memprof and callsite metadata fail verification.
; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

define i32* @test1() {
entry:
  %call1 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40), !memprof !0
  %call2 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40), !memprof !1
  %call3 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40), !memprof !3
  %call4 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40), !memprof !5
  %call5 = call noalias dereferenceable_or_null(40) i8* @malloc(i64 noundef 40), !memprof !7, !callsite !9
  %0 = bitcast i8* %call5 to i32*
  ret i32* %0
}

define i32* @test2() {
entry:
  %call = call noundef i32* @test1(), !callsite !10
  ret i32* %call
}

define i32* @test3() {
entry:
  %call = call noundef i32* @test2(), !callsite !11
  ret i32* %call
}

define void @wronginsttype() {
  %1 = add i32 0, 1, !memprof !0, !callsite !9
  ret void
}

declare dso_local noalias noundef i8* @malloc(i64 noundef)

; CHECK: !memprof annotations should have at least 1 metadata operand (MemInfoBlock)
!0 = !{}
!1 = !{!2}
; CHECK: !memprof MemInfoBlock first operand should not be null
!2 = !{null, !"cold"}
!3 = !{!4}
; CHECK: !memprof MemInfoBlock first operand should be an MDNode
!4 = !{i64 0, !"cold"}
!5 = !{!6}
; CHECK: Each !memprof MemInfoBlock should have at least 2 operands
!6 = !{i64 0}
!7 = !{!8}
; CHECK: call stack metadata should have at least 1 operand
; CHECK: Not all !memprof MemInfoBlock operands 1 to N are MDString
!8 = !{!0, !"default", i64 0}
!9 = !{i64 123}
; CHECK: call stack metadata operand should be constant integer
!10 = !{!"wrongtype"}
!11 = !{i64 789, i64 678}

; Errors from annotating incorrect instruction type in @wronginsttype.
; CHECK: !memprof metadata should only exist on calls
; CHECK: !callsite metadata should only exist on calls
