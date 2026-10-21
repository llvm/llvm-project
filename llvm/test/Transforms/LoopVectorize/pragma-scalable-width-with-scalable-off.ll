; RUN: opt -force-target-supports-scalable-vectors -scalable-vectorization=off -passes=loop-vectorize -S %s | FileCheck %s
;
; Clang lowers "#pragma clang loop vectorize_width(1, scalable)" and
; "interleave_count(1)" to width 1, scalable.enable, and interleave count 1.
; -scalable-vectorization=off overrides scalable.enable and disables vectorisation
;
; CHECK-LABEL: define void @repro(
; CHECK-NOT: vector.body:

define void @repro(ptr %out, ptr %in, i32 %tc) {
entry:
  %start = zext i32 %tc to i64
  br label %loop

loop:
  %iv = phi i64 [ %start, %entry ], [ %next, %loop ]
  %in.ptr = getelementptr inbounds i8, ptr %in, i64 %iv
  %value = load i8, ptr %in.ptr, align 1
  %sum = add i8 %value, 10
  %out.ptr = getelementptr inbounds i8, ptr %out, i64 %iv
  store i8 %sum, ptr %out.ptr, align 1
  %next = add nsw i64 %iv, -1
  %next32 = and i64 %next, 4294967295
  %done = icmp eq i64 %next32, 0
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 1}
!3 = !{!"llvm.loop.interleave.count", i32 1}
!4 = !{!"llvm.loop.vectorize.scalable.enable"}
