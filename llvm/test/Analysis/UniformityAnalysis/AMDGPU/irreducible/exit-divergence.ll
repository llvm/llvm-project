; RUN: opt %s -mtriple amdgcn-- -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

; CHECK=LABEL: UniformityInfo for function 'basic':
; CHECK-NOT: CYCLES ASSUMED DIVERGENT:
; CHECK: CYCLES WITH DIVERGENT EXIT:
; CHECK:   depth=1: entries(P T) Q
define amdgpu_kernel void @basic(i32 %a, i32 %b, i32 %c) {
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 br i1 %cond.uni, label %T, label %P

P:
  %pp.phi.1  = phi i32 [ %a, %entry], [ %b, %T ]
  %pp.phi.2  = phi i32 [ %a, %entry], [ %tt.phi, %T ]
  %pp = add i32 %b, 1
  br label %Q

Q:
  %qq = add i32 %b, 1
  %qq.div.1 = add i32 %pp.phi.2, 1
  %qq.div.2 = add i32 %pp.phi.2, 1
  br i1 %cond.div, label %T, label %exit

T:
  %tt.phi = phi i32 [ %qq, %Q ], [ %a, %entry ]
  %tt = add i32 %b, 1
  br label %P

exit:
; CHECK:   DIVERGENT:   %ee.1 =
; CHECK:   DIVERGENT:   %xx.2 =
; CHECK-NOT: DIVERGENT:     %ee.3 =
  %ee.1 = add i32 %pp.phi.1, 1
  %xx.2 = add i32 %pp.phi.2, 1
  %ee.3 = add i32 %b, 1
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'outer_reducible':
; CHECK-NOT: CYCLES ASSUMED DIVERGENT:
; CHECK: CYCLES WITH DIVERGENT EXIT:
; CHECK:   depth=1: entries(H) P T R Q
define amdgpu_kernel void @outer_reducible(i32 %a, i32 %b, i32 %c) {
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 br label %H

H:
 br i1 %cond.uni, label %T, label %P

P:
  %pp.phi.1  = phi i32 [ %a, %H], [ %b, %T ]
  %pp.phi.2  = phi i32 [ %a, %H], [ %tt.phi, %T ]
  %pp = add i32 %b, 1
  br label %Q

Q:
  %qq = add i32 %b, 1
  %qq.div.1 = add i32 %pp.phi.2, 1
  %qq.div.2 = add i32 %pp.phi.2, 1
  br i1 %cond.div, label %R, label %exit

R:
  br i1 %cond.uni, label %T, label %H


T:
  %tt.phi = phi i32 [ %qq, %R ], [ %a, %H ]
  %tt = add i32 %b, 1
  br label %P

exit:
; CHECK:   DIVERGENT:   %ee.1 =
; CHECK:   DIVERGENT:   %xx.2 =
; CHECK-NOT: DIVERGENT:     %ee.3 =
  %ee.1 = add i32 %pp.phi.1, 1
  %xx.2 = add i32 %pp.phi.2, 1
  %ee.3 = add i32 %b, 1
  ret void
}

;      entry(div)
;      |   \
;      H -> B
;      ^   /|
;      \--C |
;          \|
;           X
;
; This has a divergent cycle due to the external divergent branch, but
; there are no divergent exits. Hence a use at X is not divergent
; unless the def itself is divergent.
;
; CHECK-LABEL: UniformityInfo for function 'no_divergent_exit':
; CHECK: CYCLES ASSUMED DIVERGENT:
; CHECK:   depth=1: entries(H B) C
; CHECK-NOT: CYCLES WITH DIVERGENT EXIT:
define amdgpu_kernel void @no_divergent_exit(i32 %n, i32 %a, i32 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %div.cond, label %B, label %H

H:                                                ; preds = %C, %entry
; CHECK: DIVERGENT:  %div.merge.h =
  %div.merge.h = phi i32 [ 0, %entry ], [ %b, %C ]
  br label %B

B:                                                ; preds = %H, %entry
; CHECK: DIVERGENT:  %div.merge.b =
  %div.merge.b = phi i32 [ %a, %H ], [ 1, %entry ]
; CHECK-NOT: DIVERGENT  %bb =
  %bb = add i32 %a, 1
; CHECK-NOT: DIVERGENT:  br i1 %uni.cond, label %X, label %C
  br i1 %uni.cond, label %X, label %C

C:                                                ; preds = %B
; CHECK-NOT: DIVERGENT  %cc =
  %cc = add i32 %a, 1
; CHECK-NOT: DIVERGENT:  br i1 %uni.cond, label %X, label %H
  br i1 %uni.cond, label %X, label %H

; CHECK-LABEL: BLOCK X
X:                                                ; preds = %C, %B
; CHECK: DIVERGENT:  %uni.merge.x =
  %uni.merge.x = phi i32 [ %bb, %B ], [%cc, %C ]
; CHECK: DIVERGENT: %div.merge.x =
  %div.merge.x = phi i32 [ %div.merge.b, %B ], [%cc, %C ]
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
