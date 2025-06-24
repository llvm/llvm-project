; Test MD_prof validation

; RUN: split-file %s %t
; RUN: opt -passes=verify %t/valid.ll --disable-output
; RUN: not opt -passes=verify %t/invalid1.ll --disable-output 2>&1 | FileCheck %s
; RUN: not opt -passes=verify %t/invalid2.ll --disable-output 2>&1 | FileCheck %s

;--- valid.ll
define void @test(i1 %0) {
  br i1 %0, label %2, label %3, !prof !0
2:
  ret void
3:
  ret void
}
!0 = !{!"branch_weights", i32 1, i32 2}

;--- invalid1.ll
define void @test(i1 %0) {
  br i1 %0, label %2, label %3, !prof !0
2:
  ret void
3:
  ret void
}
!0 = !{!"invalid", i32 1, i32 2}

;--- invalid2.ll
define void @test(i1 %0) {
  br i1 %0, label %2, label %3, !prof !0
2:
  ret void
3:
  ret void
}

!0 = !{!"function_entry_count", i32 1}

; CHECK: expected either branch_weights or VP profile name
