; RUN: opt < %s -slp-vectorizer -S -mtriple=x86_64-apple-macosx10.10.0 -mattr=+sse4.2 | FileCheck %s

; CHECK-LABEL: @test
; CHECK: [[CAST:%.*]] = bitcast i32* %p to <8 x i32>*
; CHECK: [[LOAD:%.*]] = load <8 x i32>, <8 x i32>* [[CAST]], align 4
; CHECK: mul <8 x i32> <i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42>, [[LOAD]]

define i32 @test(i32* nocapture readonly %p) {
entry:
  %arrayidx.1 = getelementptr inbounds i32, i32* %p, i64 1
  %arrayidx.2 = getelementptr inbounds i32, i32* %p, i64 2
  %arrayidx.3 = getelementptr inbounds i32, i32* %p, i64 3
  %arrayidx.4 = getelementptr inbounds i32, i32* %p, i64 4
  %arrayidx.5 = getelementptr inbounds i32, i32* %p, i64 5
  %arrayidx.6 = getelementptr inbounds i32, i32* %p, i64 6
  %arrayidx.7 = getelementptr inbounds i32, i32* %p, i64 7
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %add.7, %for.body ]
  %tmp = load i32, i32* %p, align 4
  %mul = mul i32 %tmp, 42
  %add = add i32 %mul, %sum
  %tmp5 = load i32, i32* %arrayidx.1, align 4
  %mul.1 = mul i32 %tmp5, 42
  %add.1 = add i32 %mul.1, %add
  %tmp6 = load i32, i32* %arrayidx.2, align 4
  %mul.2 = mul i32 %tmp6, 42
  %add.2 = add i32 %mul.2, %add.1
  %tmp7 = load i32, i32* %arrayidx.3, align 4
  %mul.3 = mul i32 %tmp7, 42
  %add.3 = add i32 %mul.3, %add.2
  %tmp8 = load i32, i32* %arrayidx.4, align 4
  %mul.4 = mul i32 %tmp8, 42
  %add.4 = add i32 %mul.4, %add.3
  %tmp9 = load i32, i32* %arrayidx.5, align 4
  %mul.5 = mul i32 %tmp9, 42
  %add.5 = add i32 %mul.5, %add.4
  %tmp10 = load i32, i32* %arrayidx.6, align 4
  %mul.6 = mul i32 %tmp10, 42
  %add.6 = add i32 %mul.6, %add.5
  %tmp11 = load i32, i32* %arrayidx.7, align 4
  %mul.7 = mul i32 %tmp11, 42
  %add.7 = add i32 %mul.7, %add.6
  br i1 true, label %for.end, label %for.body

for.end:
  ret i32 %add.7
}
