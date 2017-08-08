; RUN: opt < %s -indvars -S | FileCheck %s

define void @test0(i32* %a) {
; CHECK-LABEL: @test0(
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %div = sdiv i32 %i.01, 2
; CHECK-NOT: sdiv
; CHECK:     udiv
  %idxprom = sext i32 %div to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  store i32 %i.01, i32* %arrayidx, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 64
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @test1(i32* %a) {
; CHECK-LABEL: @test1(
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %div = sdiv exact i32 %i.01, 2
; CHECK-NOT: sdiv
; CHECK:     udiv exact
  %idxprom = sext i32 %div to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  store i32 %i.01, i32* %arrayidx, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 64
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @test2(i32* %a, i32 %d) {
; CHECK-LABEL: @test2(
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %mul = mul nsw i32 %i.01, 64
  %div = sdiv i32 %mul, %d
; CHECK-NOT: udiv
  %idxprom = sext i32 %div to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  store i32 %i.01, i32* %arrayidx, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 64
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @test3(i32* %a) {
; CHECK-LABEL: @test3(
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %div = sdiv i32 2048, %i.01
; CHECK:     udiv
; CHECK-NOT: sdiv
  %idxprom = sext i32 %div to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  store i32 %i.01, i32* %arrayidx, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 64
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @test4(i32* %a) {
; CHECK-LABEL: @test4(
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %mul = mul nsw i32 %i.01, 64
  %div = sdiv i32 %mul, 8
; CHECK:     udiv
; CHECK-NOT: sdiv
  %idxprom = sext i32 %div to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  store i32 %i.01, i32* %arrayidx, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 64
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @test5(i32* %a) {
; CHECK-LABEL: @test5(
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %mul = mul nsw i32 %i.01, 64
  %div = sdiv i32 %mul, 6
; CHECK:     udiv
; CHECK-NOT: sdiv
  %idxprom = sext i32 %div to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  store i32 %i.01, i32* %arrayidx, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 64
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

