; RUN: opt < %s -passes='instcombine,indvars' -indvars-widen-indvars=false -S \
; RUN:   | opt -passes=loop-reduce -S | FileCheck %s

target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"

declare void @usef(float)

define void @gpu_addr_hoist(i32 %tmp20, i64 %sub47, i64 %stride, i32 %sub, ptr %tau1) {
; CHECK-LABEL: @gpu_addr_hoist(
; CHECK: call i32 @llvm.smax.i32(i32 [[SUB:%.*]], i32 0)
; CHECK: body:
; CHECK: sext i32 {{%.*}} to i64
; CHECK: mul nsw i64 [[STRIDE:%.*]], {{%.*}}
; CHECK: add nsw i64 [[SUB47:%.*]], {{%.*}}
; CHECK: shl nsw i64 {{%.*}}, 2
; CHECK: getelementptr inbounds i8, ptr [[TAU1:%.*]], i64 {{%.*}}
; CHECK: load float, ptr {{%.*}}
; CHECK: icmp ne i32 {{%.*}}, 0
entry:
  br label %pre
pre:
  br label %body
body:
  %j = phi i32 [ 0, %pre ], [ %inc, %cond ]
  %add43 = add nsw i32 %j, %tmp20
  %conv50 = sext i32 %add43 to i64
  %mul51 = mul nsw i64 %stride, %conv50
  %add52 = add nsw i64 %sub47, %mul51
  %idx = shl nsw i64 %add52, 2
  %gep = getelementptr inbounds i8, ptr %tau1, i64 %idx
  %val = load float, ptr %gep, align 4
  call void @usef(float %val)
  br label %cond
cond:
  %inc = add nuw nsw i32 %j, 1
  %cmp = icmp sgt i32 %inc, %sub
  br i1 %cmp, label %exit, label %body
exit:
  ret void
}

define void @gpu_twostream_hoist(i32 %tmp20, i64 %sub47, i64 %stride, i32 %sub, ptr %tau1, ptr %tau2) {
; CHECK-LABEL: @gpu_twostream_hoist(
; CHECK: body:
; CHECK: sext i32 {{%.*}} to i64
; CHECK: mul nsw i64 [[STRIDE:%.*]], {{%.*}}
; CHECK: getelementptr inbounds i8, ptr [[TAU1:%.*]], i64 {{%.*}}
; CHECK: getelementptr inbounds i8, ptr [[TAU2:%.*]], i64 {{%.*}}
; CHECK: load float, ptr {{%.*}}
; CHECK: load float, ptr {{%.*}}
; CHECK: icmp ne i32 {{%.*}}, 0
entry:
  br label %pre
pre:
  br label %body
body:
  %j = phi i32 [ 0, %pre ], [ %inc, %cond ]
  %add43 = add nsw i32 %j, %tmp20
  %conv50 = sext i32 %add43 to i64
  %mul51 = mul nsw i64 %stride, %conv50
  %add52 = add nsw i64 %sub47, %mul51
  %idx = shl nsw i64 %add52, 2
  %gep1 = getelementptr inbounds i8, ptr %tau1, i64 %idx
  %gep2 = getelementptr inbounds i8, ptr %tau2, i64 %idx
  %val1 = load float, ptr %gep1, align 4
  %val2 = load float, ptr %gep2, align 4
  call void @usef(float %val1)
  call void @usef(float %val2)
  br label %cond
cond:
  %inc = add nuw nsw i32 %j, 1
  %cmp = icmp sgt i32 %inc, %sub
  br i1 %cmp, label %exit, label %body
exit:
  ret void
}

define void @gpu_four_load(i32 %tmp20, i64 %sub47, i64 %stride, i32 %sub, ptr %tau1) {
; CHECK-LABEL: @gpu_four_load(
; CHECK: body:
; CHECK: sext i32 {{%.*}} to i64
; CHECK: mul nsw i64 [[STRIDE:%.*]], {{%.*}}
; CHECK: load float, ptr {{%.*}}
; CHECK: load float, ptr {{%.*}}
; CHECK: load float, ptr {{%.*}}
; CHECK: load float, ptr {{%.*}}
; CHECK: icmp ne i32 {{%.*}}, 0
entry:
  br label %pre
pre:
  br label %body
body:
  %j = phi i32 [ 0, %pre ], [ %inc, %cond ]
  %add43 = add nsw i32 %j, %tmp20
  %conv50 = sext i32 %add43 to i64
  %mul51 = mul nsw i64 %stride, %conv50
  %add52 = add nsw i64 %sub47, %mul51
  %idx = shl nsw i64 %add52, 2
  %gep = getelementptr inbounds i8, ptr %tau1, i64 %idx
  %gep1 = getelementptr inbounds i8, ptr %gep, i64 4
  %gep2 = getelementptr inbounds i8, ptr %gep, i64 8
  %gep3 = getelementptr inbounds i8, ptr %gep, i64 12
  %v0 = load float, ptr %gep, align 4
  %v1 = load float, ptr %gep1, align 4
  %v2 = load float, ptr %gep2, align 4
  %v3 = load float, ptr %gep3, align 4
  call void @usef(float %v0)
  call void @usef(float %v1)
  call void @usef(float %v2)
  call void @usef(float %v3)
  br label %cond
cond:
  %inc = add nuw nsw i32 %j, 1
  %cmp = icmp sgt i32 %inc, %sub
  br i1 %cmp, label %exit, label %body
exit:
  ret void
}
