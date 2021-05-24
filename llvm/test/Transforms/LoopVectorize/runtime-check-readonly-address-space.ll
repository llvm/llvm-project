; RUN: opt -S -march=r600 -mcpu=cayman -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine < %s | FileCheck %s

; Artificial datalayout
target datalayout = "e-p:32:32:32-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-v2048:2048:2048-n32:64"


define void @add_ints_1_1_1(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c) #0 {
; CHECK-LABEL: @add_ints_1_1_1(
; CHECK: <4 x i32>
; CHECK: ret

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %i.01
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %c, i64 %i.01
  %1 = load i32, i32 addrspace(1)* %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %i.01
  store i32 %add, i32 addrspace(1)* %arrayidx2, align 4
  %inc = add i64 %i.01, 1
  %cmp = icmp ult i64 %inc, 200
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @add_ints_as_1_0_0(i32 addrspace(1)* %a, i32* %b, i32* %c) #0 {
; CHECK-LABEL: @add_ints_as_1_0_0(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %i.01
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %c, i64 %i.01
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %i.01
  store i32 %add, i32 addrspace(1)* %arrayidx2, align 4
  %inc = add i64 %i.01, 1
  %cmp = icmp ult i64 %inc, 200
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @add_ints_as_0_1_0(i32* %a, i32 addrspace(1)* %b, i32* %c) #0 {
; CHECK-LABEL: @add_ints_as_0_1_0(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %i.01
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %c, i64 %i.01
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %i.01
  store i32 %add, i32* %arrayidx2, align 4
  %inc = add i64 %i.01, 1
  %cmp = icmp ult i64 %inc, 200
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @add_ints_as_0_1_1(i32* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c) #0 {
; CHECK-LABEL: @add_ints_as_0_1_1(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %i.01
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %c, i64 %i.01
  %1 = load i32, i32 addrspace(1)* %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %i.01
  store i32 %add, i32* %arrayidx2, align 4
  %inc = add i64 %i.01, 1
  %cmp = icmp ult i64 %inc, 200
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @add_ints_as_0_1_2(i32* %a, i32 addrspace(1)* %b, i32 addrspace(2)* %c) #0 {
; CHECK-LABEL: @add_ints_as_0_1_2(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %i.01
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(2)* %c, i64 %i.01
  %1 = load i32, i32 addrspace(2)* %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %i.01
  store i32 %add, i32* %arrayidx2, align 4
  %inc = add i64 %i.01, 1
  %cmp = icmp ult i64 %inc, 200
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
