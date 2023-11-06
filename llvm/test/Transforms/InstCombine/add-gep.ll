; RUN: opt < %s -S -passes=instcombine | FileCheck %s

target datalayout = "e-p:32:32"

; CHECK-LABEL: @test1
define void @test1(i32 %N, i32 %k, ptr %A) {
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i, %N
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: for.body:
; CHECK:      [[GEP:%.*]] = getelementptr inbounds i8, ptr %A, i32 %k
; CHECK-NEXT: %arrayidx = getelementptr inbounds i8, ptr [[GEP]], i32 %i
for.body:
  %add = add i32 %i, %k
  %arrayidx = getelementptr inbounds i8, ptr %A, i32 %add
  store i8 1, ptr %arrayidx, align 4
  %inc = add i32 %i, 1
  br label %for.cond

for.end:
  ret void
}

; CHECK-LABEL: @test2
define void @test2(i32 %N, i32 %k, ptr %A) {
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i, %N
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: for.body:
; CHECK:      [[GEP:%.*]] = getelementptr inbounds i8, ptr %A, i32 %mul
; CHECK-NEXT: %arrayidx = getelementptr inbounds i8, ptr [[GEP]], i32 %i
for.body:
  %mul = mul i32 %k, 42
  %add = add i32 %i, %mul
  %arrayidx = getelementptr inbounds i8, ptr %A, i32 %add
  store i8 1, ptr %arrayidx, align 4
  %inc = add i32 %i, 1
  br label %for.cond

for.end:
  ret void
}

; CHECK-LABEL: @test3
define void @test3(i32 %N, ptr %A, i32 %val) {
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 0, %entry ], [ %inc6, %for.inc5 ]
  %cmp = icmp ult i32 %i, %N
  br i1 %cmp, label %for.body, label %for.end7

for.body:
  br label %for.cond1

for.cond1:
  %j = phi i32 [ 0, %for.body ], [ %inc, %for.body3 ]
  %cmp2 = icmp ult i32 %j, %N
  br i1 %cmp2, label %for.body3, label %for.inc5

; CHECK-LABEL: for.body3:
; CHECK:      [[GEP:%.*]] = getelementptr inbounds i8, ptr %A, i32 %i
; CHECK-NEXT: %arrayidx = getelementptr inbounds i8, ptr [[GEP]], i32 %j
for.body3:
  %add = add i32 %i, %j
  %arrayidx = getelementptr inbounds i8, ptr %A, i32 %add
  store i8 1, ptr %arrayidx, align 4
  %inc = add i32 %j, 1
  br label %for.cond1

for.inc5:
  %inc6 = add i32 %i, 1
  br label %for.cond

for.end7:
  ret void
}

; CHECK-LABEL: @test4
define void @test4(i32 %N, ptr %A, i32 %val) {
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 0, %entry ], [ %inc6, %for.inc5 ]
  %cmp = icmp ult i32 %i, %N
  br i1 %cmp, label %for.body, label %for.end7

for.body:
  br label %for.cond1

for.cond1:
  %j = phi i32 [ 0, %for.body ], [ %inc, %for.body3 ]
  %cmp2 = icmp ult i32 %j, %N
  br i1 %cmp2, label %for.body3, label %for.inc5

; CHECK-LABEL: for.body3:
; CHECK:      [[GEP:%.*]] = getelementptr inbounds i8, ptr %A, i32 %mul
; CHECK-NEXT: %arrayidx = getelementptr inbounds i8, ptr [[GEP]], i32 %j
for.body3:
  %mul = mul i32 %i, %N
  %add = add i32 %mul, %j
  %arrayidx = getelementptr inbounds i8, ptr %A, i32 %add
  store i8 1, ptr %arrayidx, align 4
  %inc = add i32 %j, 1
  br label %for.cond1

for.inc5:
  %inc6 = add i32 %i, 1
  br label %for.cond

for.end7:
  ret void
}

; We can't use inbounds here because the add operand doesn't dominate the loop
; CHECK-LABEL: @test5
define void @test5(i32 %N, ptr %A, ptr %B) {
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i, %N
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: for.body:
; CHECK:      [[GEP:%.*]] = getelementptr i8, ptr %A, i32 %i
; CHECK-NEXT: %arrayidx = getelementptr i8, ptr [[GEP]], i32 %0
for.body:
  %0 = load i32, ptr %B, align 4
  %add = add i32 %i, %0
  %arrayidx = getelementptr inbounds i8, ptr %A, i32 %add
  store i8 1, ptr %arrayidx, align 4
  %inc = add i32 %i, 1
  br label %for.cond

for.end:
  ret void
}

; We can't use inbounds here because we don't have a loop
; CHECK-LABEL: @test6
define void @test6(i32 %k, i32 %j, ptr %A) {
entry:
  %cmp = icmp ugt i32 %k, 10
  br i1 %cmp, label %if.then, label %if.else

if.then:
  br label %if.end

if.else:
  br label %if.end

; CHECK-LABEL: if.end:
; CHECK:      [[GEP:%.*]] = getelementptr i8, ptr %A, i32 %val
; CHECK-NEXT: %arrayidx = getelementptr i8, ptr [[GEP]], i32 %j
if.end:
  %val = phi i32 [ 0, %if.then ], [ 1, %if.else ]
  %add = add i32 %val, %j
  %arrayidx = getelementptr inbounds i8, ptr %A, i32 %add
  store i8 1, ptr %arrayidx, align 4
  ret void
}

; Inbounds gep would be invalid because of potential overflow in the add, though
; we don't convert to gep+gep as we insert an explicit sext instead of using i16
; gep offset.
; CHECK-LABEL: @test7
define void @test7(i16 %N, i16 %k, ptr %A) {
entry:
  br label %for.cond

for.cond:
  %i = phi i16 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i16 %i, %N
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: for.body:
; CHECK:      %add = add i16 %i, %k
; CHECK-NEXT: [[SEXT:%.*]] = sext i16 %add to i32
; CHECK-NEXT: %arrayidx = getelementptr inbounds i8, ptr %A, i32 [[SEXT]]
for.body:
  %add = add i16 %i, %k
  %arrayidx = getelementptr inbounds i8, ptr %A, i16 %add
  store i8 1, ptr %arrayidx, align 4
  %inc = add i16 %i, 1
  br label %for.cond

for.end:
  ret void
}

; %i starts at 1 so we can't use inbounds
; CHECK-LABEL: @test8
define void @test8(i32 %N, i32 %k, ptr %A) {
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 1, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i, %N
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: for.body:
; CHECK:      [[GEP:%.*]] = getelementptr i8, ptr %A, i32 %i
; CHECK-NEXT: %arrayidx = getelementptr i8, ptr [[GEP]], i32 %k
for.body:
  %add = add i32 %i, %k
  %arrayidx = getelementptr inbounds i8, ptr %A, i32 %add
  store i8 1, ptr %arrayidx, align 4
  %inc = add i32 %i, 1
  br label %for.cond

for.end:
  ret void
}
