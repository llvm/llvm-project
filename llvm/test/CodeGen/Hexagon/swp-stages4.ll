; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 -enable-pipeliner -pipeliner-max-stages=2 -disable-block-placement=0 -hexagon-bit=0 < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that we rename registers correctly for multiple stages when there is a
; Phi and depends upon another Phi.

; CHECK: = and
; CHECK: = and
; CHECK: r[[REGA:[0-9]+]] = memub(r{{[0-9]+}}+#1)
; CHECK: = and
; CHECK: r[[REG0:[0-9]+]] = and(r[[REG1:[0-9]+]],#255)
; CHECK-NOT: r[[REG0]] = and(r[[REG1]],#255)
; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: = add(r{{[0-9]+}},r[[REG0]])
; CHECK: = and
; CHECK: r[[REG0]] = and
; CHECK: endloop

; Function Attrs: nounwind
define void @test(ptr noalias nocapture %src, i32 %srcWidth, i32 %srcHeight, i32 %srcStride, ptr noalias nocapture %dst, i32 %dstStride) #0 {
entry:
  %sub = add i32 %srcWidth, -1
  %sub1 = add i32 %srcHeight, -1
  %add.ptr = getelementptr inbounds i8, ptr %src, i32 %srcStride
  %add.ptr.sum = mul i32 %srcStride, 2
  %add.ptr2 = getelementptr inbounds i8, ptr %src, i32 %add.ptr.sum
  br label %for.body.lr.ph

for.body.lr.ph:
  %0 = add i32 %srcHeight, -2
  %1 = mul i32 %0, %dstStride
  %2 = mul i32 %0, %srcStride
  %3 = mul i32 %sub1, %srcStride
  br label %for.cond

for.cond:
  %scevgep = getelementptr i8, ptr %dst, i32 %1
  %scevgep220 = getelementptr i8, ptr %src, i32 %2
  %scevgep221 = getelementptr i8, ptr %src, i32 %3
  %arrayidx6 = getelementptr inbounds i8, ptr %src, i32 1
  %add11 = add i32 %srcStride, 1
  %arrayidx12 = getelementptr inbounds i8, ptr %src, i32 %add11
  br label %for.body75.preheader

for.body75.preheader:
  %sri = load i8, ptr %arrayidx6, align 1
  %sri224 = load i8, ptr %src, align 1
  %sri227 = load i8, ptr %arrayidx12, align 1
  %sri229 = load i8, ptr %add.ptr, align 1
  br label %for.body75

for.body75:
  %j.0211 = phi i32 [ %add82, %for.body75 ], [ 1, %for.body75.preheader ]
  %sr = phi i8 [ %4, %for.body75 ], [ %sri, %for.body75.preheader ]
  %sr225 = phi i8 [ %sr, %for.body75 ], [ %sri224, %for.body75.preheader ]
  %sr230 = phi i8 [ %5, %for.body75 ], [ %sri227, %for.body75.preheader ]
  %sr231 = phi i8 [ %sr230, %for.body75 ], [ %sri229, %for.body75.preheader ]
  %conv78 = zext i8 %sr225 to i32
  %conv80 = zext i8 %sr to i32
  %add81 = add nsw i32 %conv80, %conv78
  %add82 = add i32 %j.0211, 1
  %arrayidx83 = getelementptr inbounds i8, ptr %src, i32 %add82
  %4 = load i8, ptr %arrayidx83, align 1, !tbaa !0
  %conv84 = zext i8 %4 to i32
  %add85 = add nsw i32 %add81, %conv84
  %conv88 = zext i8 %sr231 to i32
  %add89 = add nsw i32 %add85, %conv88
  %conv91 = zext i8 %sr230 to i32
  %add92 = add nsw i32 %add89, %conv91
  %add.ptr.sum208 = add i32 %add82, %srcStride
  %arrayidx94 = getelementptr inbounds i8, ptr %src, i32 %add.ptr.sum208
  %5 = load i8, ptr %arrayidx94, align 1, !tbaa !0
  %conv95 = zext i8 %5 to i32
  %add96 = add nsw i32 %add92, %conv95
  %mul97 = mul nsw i32 %add96, 7282
  %add98 = add nsw i32 %mul97, 32768
  %shr99209 = lshr i32 %add98, 16
  %conv100 = trunc i32 %shr99209 to i8
  %arrayidx101 = getelementptr inbounds i8, ptr %dst, i32 %j.0211
  store i8 %conv100, ptr %arrayidx101, align 1, !tbaa !0
  %exitcond = icmp eq i32 %add82, %sub
  br i1 %exitcond, label %for.end104.loopexit, label %for.body75

for.end104.loopexit:
  br label %for.end104

for.end104:
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!"omnipotent char", !1}
!1 = !{!"Simple C/C++ TBAA"}
