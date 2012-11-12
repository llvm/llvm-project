; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s
;rdar://8003725

@G1 = external global i32
@G2 = external global i32

define i32 @f1(i32 %cond1, i32 %x1, i32 %x2, i32 %x3) {
entry:
; CHECK: f1:
; CHECK: cmp
; CHECK: moveq
; CHECK-NOT: cmp
; CHECK: mov{{eq|ne}}
    %tmp1 = icmp eq i32 %cond1, 0
    %tmp2 = select i1 %tmp1, i32 %x1, i32 %x2
    %tmp3 = select i1 %tmp1, i32 %x2, i32 %x3
    %tmp4 = add i32 %tmp2, %tmp3
    ret i32 %tmp4
}

@foo = external global i32
@bar = external global [250 x i8], align 1

; CSE of cmp across BB boundary
; rdar://10660865
define void @f2() nounwind ssp {
entry:
; CHECK: f2:
; CHECK: cmp
; CHECK: poplt
; CHECK-NOT: cmp
; CHECK: movle
  %0 = load i32* @foo, align 4
  %cmp28 = icmp sgt i32 %0, 0
  br i1 %cmp28, label %for.body.lr.ph, label %for.cond1.preheader

for.body.lr.ph:                                   ; preds = %entry
  %1 = icmp sgt i32 %0, 1
  %smax = select i1 %1, i32 %0, i32 1
  call void @llvm.memset.p0i8.i32(i8* getelementptr inbounds ([250 x i8]* @bar, i32 0, i32 0), i8 0, i32 %smax, i32 1, i1 false)
  unreachable

for.cond1.preheader:                              ; preds = %entry
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind
