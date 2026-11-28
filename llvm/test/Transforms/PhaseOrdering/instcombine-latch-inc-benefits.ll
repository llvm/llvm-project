; SCEV gains from preserving post-inc latch exit compares

; RUN: opt < %s -passes='instcombine,print<scalar-evolution>' -disable-output 2>&1 \
; RUN:   | FileCheck %s
; RUN: opt < %s -passes='instcombine,indvars,print<scalar-evolution>' \
; RUN:   -indvars-widen-indvars=false -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=INDVARS

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"

@a = external global [1024 x i32]
declare void @use(i32)
declare void @usef(float)

define void @scev_postinc_nsw(i32 %sub) {
; CHECK-LABEL: 'scev_postinc_nsw'
; CHECK: %inc = add nuw nsw i32 %j, 1
; CHECK-NEXT: --> {1,+,1}<nuw><nsw>
entry:
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %inc = add nuw nsw i32 %j, 1
  %cmp = icmp sgt i32 %inc, %sub
  br i1 %cmp, label %exit, label %loop
exit:
  ret void
}

define void @scev_gep_nsw(i32 %sub) {
; CHECK-LABEL: 'scev_gep_nsw'
; CHECK: %inc = add nuw nsw i32 %j, 1
; CHECK-NEXT: --> {1,+,1}<nuw><nsw>
entry:
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %idx = zext i32 %j to i64
  %off = mul nuw nsw i64 %idx, 4
  %p = getelementptr i8, ptr @a, i64 %off
  %v = load i32, ptr %p
  %inc = add nuw nsw i32 %j, 1
  %cmp = icmp sgt i32 %inc, %sub
  br i1 %cmp, label %exit, label %loop
exit:
  ret void
}

define void @scev_gpu_nssw(i32 %tmp20, i64 %sub47, i64 %stride, i32 %sub, ptr %tau1) {
; CHECK-LABEL: 'scev_gpu_nssw'
; CHECK: %conv50 = sext i32 %add43 to i64
; CHECK-NEXT: --> {(sext i32 %tmp20 to i64),+,1}<nsw>
; CHECK: {1,+,1}<nuw><{{.*}}> Added Flags: <nssw>
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
  %gep.idx = shl nsw i64 %add52, 2
  %gep = getelementptr inbounds i8, ptr %tau1, i64 %gep.idx
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

define void @scev_nssw_indvars(i32 %sub) {
; INDVARS-LABEL: 'scev_nssw_indvars'
; INDVARS: {1,+,1}<nuw><{{.*}}> Added Flags: <nssw>
entry:
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %inc, %loop ]
  call void @use(i32 %j)
  %inc = add nuw nsw i32 %j, 1
  %cmp = icmp sgt i32 %inc, %sub
  br i1 %cmp, label %exit, label %loop
exit:
  ret void
}
