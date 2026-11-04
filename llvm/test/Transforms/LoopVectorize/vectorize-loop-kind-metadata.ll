; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 \
; RUN:   -pass-remarks-analysis=loop-vectorize -S | FileCheck %s --check-prefix=REMARKS
; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S \
; RUN:   | FileCheck %s --check-prefix=NO-REMARKS

; Verify that llvm.loop.vectorize.body is attached to the vector loop and
; llvm.loop.vectorize.epilogue is attached to the scalar remainder when
; optimization remarks are enabled, and absent otherwise.

define void @test_metadata(ptr noalias %A, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %A, i64 %iv
  %val = load i32, ptr %gep, align 4
  %add = add i32 %val, 1
  store i32 %add, ptr %gep, align 4
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; With remarks enabled: vector loop has vectorize.body, scalar remainder has vectorize.epilogue.
; REMARKS-DAG: !{!"llvm.loop.vectorize.body", i32 1}
; REMARKS-DAG: !{!"llvm.loop.vectorize.epilogue", i32 1}

; Without remarks: neither metadata is present.
; NO-REMARKS-NOT: llvm.loop.vectorize.body
; NO-REMARKS-NOT: llvm.loop.vectorize.epilogue
