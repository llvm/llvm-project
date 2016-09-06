; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -indvars %s | FileCheck %s

; Bug 21148

; Induction variables should not be widened for 64-bit integers,
; despite being a legal type.
;
; The cost of basic arithmetic instructions on a 64-bit integer are
; twice as expensive as that on a 32-bit integer, or split into 2
; 32-bit components.

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

; CHECK-LABEL: @indvar_32_bit(
; CHECK-NOT: sext i32
; CHECK: phi i32
define void @indvar_32_bit(i32 %n, i32* nocapture %output) {
entry:
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i32 [ 0, %for.body.preheader ], [ %add, %for.body ]
  %mul = mul nsw i32 %i.06, %i.06
  %tmp0 = sext i32 %i.06 to i64
  %arrayidx = getelementptr inbounds i32, i32* %output, i64 %tmp0
  store i32 %mul, i32* %arrayidx, align 4
  %add = add nsw i32 %i.06, 3
  %cmp = icmp slt i32 %add, %n
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; CHECK-LABEL: @no_promote_i32(
; CHECK-NOT: sext i32
; CHECK: br
; CHECK-NOT: shl i64
; CHECK-NOT: ashr i64
; CHECK-NOT: mul nsw i64
; CHECK-NOT: add nsw i64
define void @no_promote_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  br label %for.body

for.body:
  %inc = phi i32 [ 0, %entry ], [ %inc.i, %for.body ]
  %tmp0 = add i32 %a, %inc
  %shl = shl i32 %inc, 8
  %shr = ashr exact i32 %shl, 8
  %mul = mul nsw i32 %shr, %a
  %add = add nsw i32 %mul, %b
  %tmp1 = sext i32 %add to i64
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tmp1
  store i32 %tmp0, i32 addrspace(1)* %arrayidx1, align 4
  %inc.i = add nsw i32 %inc, 1
  %cmp = icmp slt i32 %inc.i, 16
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; FIXME: This should really be promoted to i64, since it will need to
; be legalized anyway.

; CHECK-LABEL: @indvar_48_bit(
define void @indvar_48_bit(i48 %n, i48* nocapture %output) {
entry:
  %cmp5 = icmp sgt i48 %n, 0
  br i1 %cmp5, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i48 [ 0, %for.body.preheader ], [ %add, %for.body ]
  %mul = mul nsw i48 %i.06, %i.06
  %tmp0 = sext i48 %i.06 to i64
  %arrayidx = getelementptr inbounds i48, i48* %output, i64 %tmp0
  store i48 %mul, i48* %arrayidx, align 4
  %add = add nsw i48 %i.06, 3
  %cmp = icmp slt i48 %add, %n
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
