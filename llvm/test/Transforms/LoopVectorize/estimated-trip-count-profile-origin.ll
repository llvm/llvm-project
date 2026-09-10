; RUN: split-file %s %t
; RUN: opt < %t/count.ll -passes=loop-vectorize \
; RUN:     -loop-vectorize-with-block-frequency -force-vector-width=4 \
; RUN:     -force-vector-interleave=1 -S \
; RUN:   | FileCheck %s --check-prefix=COUNT \
; RUN:       --implicit-check-not=approxprofile
; RUN: opt < %t/expected.ll -passes=loop-vectorize \
; RUN:     -loop-vectorize-with-block-frequency -force-vector-width=4 \
; RUN:     -force-vector-interleave=1 -S \
; RUN:   | FileCheck %s --check-prefix=EXPECTED \
; RUN:       --implicit-check-not=approxprofile
; RUN: opt < %t/missing.ll -passes=loop-vectorize \
; RUN:     -loop-vectorize-with-block-frequency -force-vector-width=4 \
; RUN:     -force-vector-interleave=1 -S \
; RUN:   | FileCheck %s --check-prefix=MISSING \
; RUN:       --implicit-check-not=approxprofile
;
; The vector latch gets fresh weights that are still counts, so do not stamp.
; llvm.expect stays a hint, and a latch with no !prof still gets an estimate.

; COUNT-LABEL: define void @count(
; COUNT: vector.body:
; COUNT: br i1 {{.*}}, label %middle.block, label %vector.body, !prof [[COUNT_VEC:![0-9]+]]
; COUNT: [[COUNT_VEC]] = !{!"branch_weights", i32 10, i32 2490}

; EXPECTED-LABEL: define void @expected(
; EXPECTED: vector.body:
; EXPECTED: br i1 {{.*}}, label %middle.block, label %vector.body, !prof [[VECTOR_PROF:![0-9]+]]
; EXPECTED: loop:
; EXPECTED: br i1 {{.*}}, label %exit, label %loop, !prof [[SCALAR_PROF:![0-9]+]]
; EXPECTED: [[VECTOR_PROF]] = !{!"branch_weights", !"expected", i32 10, i32 2490}
; EXPECTED: [[SCALAR_PROF]] = !{!"branch_weights", !"expected",

; MISSING-LABEL: define void @missing(
; MISSING: vector.body:
; MISSING: br i1 {{.*}}, label %middle.block, label %vector.body, !prof

;--- count.ll
define void @count(ptr %a, i32 %bound) !prof !0 {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %loop ]
  %gep = getelementptr inbounds i32, ptr %a, i32 %iv
  store i32 %iv, ptr %gep, align 4
  %next = add nuw nsw i32 %iv, 1
  %done = icmp eq i32 %next, %bound
  br i1 %done, label %exit, label %loop, !prof !1

exit:
  ret void
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", i32 10, i32 10000}

;--- expected.ll
define void @expected(ptr %a, i32 %bound) !prof !0 {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %loop ]
  %gep = getelementptr inbounds i32, ptr %a, i32 %iv
  store i32 %iv, ptr %gep, align 4
  %next = add nuw nsw i32 %iv, 1
  %done = icmp eq i32 %next, %bound
  br i1 %done, label %exit, label %loop, !prof !1

exit:
  ret void
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", !"expected", i32 10, i32 10000}

;--- missing.ll
define void @missing(ptr %a) !prof !0 {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %loop ]
  %gep = getelementptr inbounds i32, ptr %a, i32 %iv
  store i32 %iv, ptr %gep, align 4
  %next = add nuw nsw i32 %iv, 1
  %done = icmp eq i32 %next, 16
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

!0 = !{!"function_entry_count", i64 10}
