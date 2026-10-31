; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -force-vector-width=16 -tail-folding-policy=dont-fold-tail \
; RUN: -epilogue-tail-folding-policy=prefer-fold-tail -epilogue-vectorization-force-VF=8 \
; RUN: -pass-remarks-analysis=loop-vectorize -debug-only=loop-vectorize \
; RUN: -disable-output -mtriple=riscv64 -mattr=+v < %s 2>&1 | FileCheck %s --check-prefix=CHECK-FIXED-VF

; RUN: opt -passes=loop-vectorize -force-vector-width="vscale x 4" -tail-folding-policy=dont-fold-tail \
; RUN: -epilogue-tail-folding-policy=prefer-fold-tail -epilogue-vectorization-force-VF="vscale x 2" \
; RUN: -pass-remarks-analysis=loop-vectorize -debug-only=loop-vectorize \
; RUN: -disable-output -mtriple=riscv64 -mattr=+v < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SCALABLE_VF

define void @test_epilogue_tf(ptr noalias %a, ptr noalias %b, i64 %n) {
; CHECK-FIXED-VF-LABEL: Checking a loop in 'test_epilogue_tf'
; CHECK-FIXED-VF: remark: <unknown>:0:0: Epilogue tail-folding is not supported yet with EVL-based tail-folding

; CHECK-SCALABLE_VF-LABEL: Checking a loop in 'test_epilogue_tf'
; CHECK-SCALABLE_VF: remark: <unknown>:0:0: Epilogue tail-folding is not supported yet with EVL-based tail-folding

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr i32, ptr %a, i64 %iv
  %l = load i32, ptr %gep.a, align 4
  %gep.b = getelementptr i32, ptr %b, i64 %iv
  store i32 %l, ptr %gep.b, align 4
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
