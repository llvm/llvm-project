; RUN: opt -indvars -S < %s | FileCheck %s

define void @test0(i32* %x) {
entry:
  br label %for.inc

for.inc:                                          ; preds = %for.inc, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %and = and i32 %i.01, 3
  %cmp1 = icmp eq i32 %and, 0
  %cond = select i1 %cmp1, i32 0, i32 1
  store i32 %cond, i32* %x, align 4
  %add = add i32 %i.01, 4
  %cmp = icmp ult i32 %add, 8
  br i1 %cmp, label %for.inc, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

; Should fold the condition of the select into constant
; CHECK-LABEL: void @test0(
; CHECK:         icmp eq i32 0, 0

define void @test1(i32* %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %mul = mul nsw i32 %i.01, 64
  %rem = srem i32 %mul, 8
  %idxprom = sext i32 %rem to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  store i32 %i.01, i32* %arrayidx, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 64
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

; Should fold the rem since %mul is multiple of 8
; CHECK-LABEL: @test1(
; CHECK-NOT:     rem
; CHECK:         sext i32 0 to i64
