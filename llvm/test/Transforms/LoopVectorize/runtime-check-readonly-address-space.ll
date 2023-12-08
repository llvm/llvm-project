; RUN: opt -S -march=r600 -mcpu=cayman -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 < %s | FileCheck %s

; Artificial datalayout
target datalayout = "e-p:32:32:32-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-v2048:2048:2048-n32:64"


define void @add_ints_1_1_1(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) #0 {
; CHECK-LABEL: @add_ints_1_1_1(
; CHECK: <4 x i32>
; CHECK: ret

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %i.01
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(1) %c, i64 %i.01
  %1 = load i32, ptr addrspace(1) %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %i.01
  store i32 %add, ptr addrspace(1) %arrayidx2, align 4
  %inc = add i64 %i.01, 1
  %cmp = icmp ult i64 %inc, 200
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @add_ints_as_1_0_0(ptr addrspace(1) %a, ptr %b, ptr %c) #0 {
; CHECK-LABEL: @add_ints_as_1_0_0(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %i.01
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %c, i64 %i.01
  %1 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %i.01
  store i32 %add, ptr addrspace(1) %arrayidx2, align 4
  %inc = add i64 %i.01, 1
  %cmp = icmp ult i64 %inc, 200
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @add_ints_as_0_1_0(ptr %a, ptr addrspace(1) %b, ptr %c) #0 {
; CHECK-LABEL: @add_ints_as_0_1_0(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %i.01
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %c, i64 %i.01
  %1 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %i.01
  store i32 %add, ptr %arrayidx2, align 4
  %inc = add i64 %i.01, 1
  %cmp = icmp ult i64 %inc, 200
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @add_ints_as_0_1_1(ptr %a, ptr addrspace(1) %b, ptr addrspace(1) %c) #0 {
; CHECK-LABEL: @add_ints_as_0_1_1(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %i.01
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(1) %c, i64 %i.01
  %1 = load i32, ptr addrspace(1) %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %i.01
  store i32 %add, ptr %arrayidx2, align 4
  %inc = add i64 %i.01, 1
  %cmp = icmp ult i64 %inc, 200
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define void @add_ints_as_0_1_2(ptr %a, ptr addrspace(1) %b, ptr addrspace(2) %c) #0 {
; CHECK-LABEL: @add_ints_as_0_1_2(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %i.01
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(2) %c, i64 %i.01
  %1 = load i32, ptr addrspace(2) %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %i.01
  store i32 %add, ptr %arrayidx2, align 4
  %inc = add i64 %i.01, 1
  %cmp = icmp ult i64 %inc, 200
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
