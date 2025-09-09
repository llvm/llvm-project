; Test MD_prof validation

; RUN: split-file %s %t

; RUN: opt -passes=verify %t/valid.ll --disable-output

; RUN: not opt -passes=verify %t/wrong-count.ll --disable-output 2>&1 | FileCheck %s --check-prefix=WRONG-COUNT
; RUN: not opt -passes=verify %t/invalid-name1.ll --disable-output 2>&1 | FileCheck %s
; RUN: not opt -passes=verify %t/invalid-name2.ll --disable-output 2>&1 | FileCheck %s

; RUN: opt -passes=verify %t/unknown-correct.ll --disable-output

; RUN: not opt -passes=verify %t/unknown-invalid.ll --disable-output 2>&1 | FileCheck %s --check-prefix=EXTRA-ARGS
; RUN: opt -passes=verify %t/unknown-on-function1.ll -S -o - | FileCheck %s --check-prefix=ON-FUNCTION1
; RUN: not opt -passes=verify %t/unknown-on-function2.ll --disable-output 2>&1 | FileCheck %s --check-prefix=ON-FUNCTION2
; RUN: not opt -passes=verify %t/invalid-unknown-placement.ll --disable-output 2>&1 | FileCheck %s --check-prefix=INVALID-UNKNOWN-PLACEMENT

;--- valid.ll
declare void @to_invoke()
declare i32 @__gxx_personality_v0(...)

define void @invoker() personality ptr @__gxx_personality_v0 {
  invoke void @to_invoke() to label %exit unwind label %lpad, !prof !0
lpad:
  %ll = landingpad {ptr, i32}
  cleanup
  ret void
exit:
  ret void
}

define i32 @test(i32 %a) {
  %c = icmp eq i32 %a, 0
  br i1 %c, label %yes, label %exit, !prof !0
yes:
  switch i32 %a, label %exit [ i32 1, label %case_b
                               i32 2, label %case_c], !prof !1
case_b:
  br label %exit
case_c:
  br label %exit
exit:
  %r = select i1 %c, i32 1, i32 2, !prof !0
  ret i32 %r
}
!0 = !{!"branch_weights", i32 1, i32 2}
!1 = !{!"branch_weights", i32 1, i32 2, i32 3}

;--- wrong-count.ll
define void @test(i32 %a) {
  %c = icmp eq i32 %a, 0
  br i1 %c, label %yes, label %no, !prof !0
yes:
  ret void
no:
  ret void
}
!0 = !{!"branch_weights", i32 1, i32 2, i32 3}

; WRONG-COUNT: Wrong number of operands

;--- invalid-name1.ll
define void @test(i1 %0) {
  br i1 %0, label %2, label %3, !prof !0
2:
  ret void
3:
  ret void
}
!0 = !{!"invalid", i32 1, i32 2}

;--- invalid-name2.ll
define void @test(i1 %0) {
  br i1 %0, label %2, label %3, !prof !0
2:
  ret void
3:
  ret void
}

!0 = !{!"function_entry_count", i32 1}

; CHECK: expected either branch_weights or VP profile name

;--- unknown-correct.ll
declare void @to_invoke()
declare i32 @__gxx_personality_v0(...)

define void @invoker() personality ptr @__gxx_personality_v0 {
  invoke void @to_invoke() to label %exit unwind label %lpad, !prof !0
lpad:
  %ll = landingpad {ptr, i32}
  cleanup
  ret void
exit:
  ret void
}

define i32 @test(i32 %a) {
  %c = icmp eq i32 %a, 0
  br i1 %c, label %yes, label %exit, !prof !0
yes:
  switch i32 %a, label %exit [ i32 1, label %case_b
                               i32 2, label %case_c], !prof !0
case_b:
  br label %exit
case_c:
  br label %exit
exit:
  %r = select i1 %c, i32 1, i32 2, !prof !0
  ret i32 %r
}

!0 = !{!"unknown"}

;--- unknown-invalid.ll
define void @test(i32 %a) {
  %c = icmp eq i32 %a, 0
  br i1 %c, label %yes, label %no, !prof !0
yes:
  ret void
no:
  ret void
}

!0 = !{!"unknown", i32 12, i32 67}
; EXTRA-ARGS: 'unknown' !prof should have no additional operands

;--- unknown-on-function1.ll
define void @test() !prof !0 {
  ret void
}

!0 = !{!"unknown"}
; ON-FUNCTION1: define void @test() !prof !0

;--- unknown-on-function2.ll
define void @test() !prof !0 {
  ret void
}

!0 = !{!"unknown", i64 123}
; ON-FUNCTION2: first operand should be 'function_entry_count' or 'synthetic_function_entry_count'

;--- invalid-unknown-placement.ll
define i32 @test() {
  %r = add i32 1, 2, !prof !0
  ret i32 %r
}
!0 = !{!"unknown"}
; INVALID-UNKNOWN-PLACEMENT: 'unknown' !prof should only appear on instructions on which 'branch_weights' would
