; RUN: opt < %s -S -passes=loop-unroll -pass-remarks=loop-unroll -unroll-count=16 2>&1 | FileCheck -check-prefix=COMPLETE-UNROLL %s
; RUN: opt < %s -S -passes=loop-unroll -pass-remarks=loop-unroll -unroll-count=4 2>&1 | FileCheck -check-prefix=PARTIAL-UNROLL %s
; RUN: opt < %s -S -passes=loop-unroll -pass-remarks=loop-unroll -unroll-count=4 -unroll-runtime=true -unroll-remainder 2>&1 | FileCheck %s --check-prefix=RUNTIME-UNROLL
; RUN: opt < %s -S -passes=loop-unroll -pass-remarks-missed=loop-unroll 2>&1 | FileCheck -check-prefix=VECTORIZED %s

; COMPLETE-UNROLL: remark: {{.*}}: completely unrolled loop with 16 iterations
; PARTIAL-UNROLL: remark: {{.*}}: unrolled loop by a factor of 4
; RUNTIME-UNROLL: remark: {{.*}}: unrolled loop by a factor of 4

define i32 @sum() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %s.06 = phi i32 [ 0, %entry ], [ %add1, %for.body ]
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nsw i32 %i.05, 4
  %call = tail call i32 @baz(i32 %add) #2
  %add1 = add nsw i32 %call, %s.06
  %inc = add nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 16
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 %add1
}

; RUNTIME-UNROLL-NOT: remark: {{.*}}: completely unrolled loop with 3 iterations
; RUNTIME-UNROLL: remark: {{.*}}: unrolled loop by a factor of 4

define i32 @runtime(i32 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %s.06 = phi i32 [ 0, %entry ], [ %add1, %for.body ]
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nsw i32 %i.05, 4
  %call = tail call i32 @baz(i32 %add) #2
  %add1 = add nsw i32 %call, %s.06
  %inc = add nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 %add1
}

declare i32 @baz(i32)

; Vectorized loop with vector instructions — expect "already vectorized" remark.
; VECTORIZED: remark: {{.*}}: loop not unrolled: already vectorized
define void @vectorized_loop(ptr noalias %a, ptr noalias %b, i64 %n) {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds float, ptr %a, i64 %index
  %wide.load = load <4 x float>, ptr %gep.a, align 4
  %gep.b = getelementptr inbounds float, ptr %b, i64 %index
  %wide.load2 = load <4 x float>, ptr %gep.b, align 4
  %add = fadd <4 x float> %wide.load, %wide.load2
  store <4 x float> %add, ptr %gep.a, align 4
  %index.next = add nuw i64 %index, 4
  %cmp = icmp eq i64 %index.next, %n
  br i1 %cmp, label %exit, label %vector.body, !llvm.loop !0

exit:
  ret void
}

; Interleaved loop (interleave count > 1) — expect "interleaved" remark.
; VECTORIZED: remark: {{.*}}: loop not unrolled: interleaved by the vectorizer with interleave count 4
define void @interleaved_loop(ptr noalias %a, ptr noalias %b, i64 %n) {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds float, ptr %a, i64 %index
  %wide.load = load <4 x float>, ptr %gep.a, align 4
  %gep.b = getelementptr inbounds float, ptr %b, i64 %index
  %wide.load2 = load <4 x float>, ptr %gep.b, align 4
  %add = fadd <4 x float> %wide.load, %wide.load2
  store <4 x float> %add, ptr %gep.a, align 4
  %index.next = add nuw i64 %index, 16
  %cmp = icmp eq i64 %index.next, %n
  br i1 %cmp, label %exit, label %vector.body, !llvm.loop !2

exit:
  ret void
}

; Scalar remainder loop (isvectorized but no vector instructions) — no remark.
; VECTORIZED-NOT: remark: {{.*}} scalar_remainder
define void @scalar_remainder(ptr noalias %a, ptr noalias %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %gep.a = getelementptr inbounds float, ptr %a, i64 %i
  %val.a = load float, ptr %gep.a, align 4
  %gep.b = getelementptr inbounds float, ptr %b, i64 %i
  %val.b = load float, ptr %gep.b, align 4
  %add = fadd float %val.a, %val.b
  store float %add, ptr %gep.a, align 4
  %i.next = add nuw i64 %i, 1
  %cmp = icmp eq i64 %i.next, %n
  br i1 %cmp, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !4}
!2 = distinct !{!2, !4, !5}
!4 = !{!"llvm.loop.isvectorized", i32 1}
!5 = !{!"llvm.loop.interleave.count", i32 4}
