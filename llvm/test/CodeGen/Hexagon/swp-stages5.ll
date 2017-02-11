; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner -pipeliner-max-stages=2 -hexagon-bit=0 < %s | FileCheck %s

; Very similar to swp-stages4.ll, but the pipelined schedule is a little
; different.

; CHECK: = memub(r{{[0-9]+}}++#1)
; CHECK-DAG: [[REG0:(r[0-9]+)]] = memub(r{{[0-9]+}}++#1)
; CHECK-DAG: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: = and([[REG0]],#255)
; CHECK: [[REG0]]{{[:0-9]*}} =
; CHECK: endloop

define void @fred(i8* noalias nocapture %src, i32 %srcWidth, i32 %srcHeight, i32 %srcStride, i8* noalias nocapture %dst, i32 %dstStride) #0 {
entry:
  %sub = add i32 %srcWidth, -1
  %sub1 = add i32 %srcHeight, -1
  %add.ptr = getelementptr inbounds i8, i8* %src, i32 %srcStride
  %add.ptr.sum = mul i32 %srcStride, 2
  %add.ptr2 = getelementptr inbounds i8, i8* %src, i32 %add.ptr.sum
  %cmp212 = icmp ugt i32 %sub1, 1
  br i1 %cmp212, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body74.preheader

for.body74.preheader:
  %0 = load i8, i8* %add.ptr, align 1, !tbaa !0
  %arrayidx40 = getelementptr inbounds i8, i8* %add.ptr, i32 1
  %1 = load i8, i8* %arrayidx40, align 1, !tbaa !0
  %2 = load i8, i8* %add.ptr, align 1, !tbaa !0
  %arrayidx46 = getelementptr inbounds i8, i8* %add.ptr, i32 1
  %3 = load i8, i8* %arrayidx46, align 1, !tbaa !0
  br label %for.body74

for.body74:
  %4 = phi i8 [ %9, %for.body74 ], [ %3, %for.body74.preheader ]
  %5 = phi i8 [ %4, %for.body74 ], [ %2, %for.body74.preheader ]
  %6 = phi i8 [ %8, %for.body74 ], [ %1, %for.body74.preheader ]
  %7 = phi i8 [ %6, %for.body74 ], [ %0, %for.body74.preheader ]
  %j.0211 = phi i32 [ %add81, %for.body74 ], [ 1, %for.body74.preheader ]
  %conv77 = zext i8 %7 to i32
  %conv79 = zext i8 %6 to i32
  %add80 = add nsw i32 %conv79, %conv77
  %add81 = add i32 %j.0211, 1
  %arrayidx82 = getelementptr inbounds i8, i8* %src, i32 %add81
  %8 = load i8, i8* %arrayidx82, align 1, !tbaa !0
  %conv83 = zext i8 %8 to i32
  %add84 = add nsw i32 %add80, %conv83
  %conv87 = zext i8 %5 to i32
  %add88 = add nsw i32 %add84, %conv87
  %conv90 = zext i8 %4 to i32
  %add91 = add nsw i32 %add88, %conv90
  %arrayidx93 = getelementptr inbounds i8, i8* %add.ptr, i32 %add81
  %9 = load i8, i8* %arrayidx93, align 1, !tbaa !0
  %conv94 = zext i8 %9 to i32
  %add95 = add nsw i32 %add91, %conv94
  %mul96 = mul nsw i32 %add95, 7282
  %add97 = add nsw i32 %mul96, 32768
  %shr98208 = lshr i32 %add97, 16
  %conv99 = trunc i32 %shr98208 to i8
  %add.ptr5.sum209 = add i32 %j.0211, %dstStride
  %arrayidx100 = getelementptr inbounds i8, i8* %dst, i32 %add.ptr5.sum209
  store i8 %conv99, i8* %arrayidx100, align 1, !tbaa !0
  %exitcond = icmp eq i32 %add81, %sub
  br i1 %exitcond, label %for.end103.loopexit, label %for.body74

for.end103.loopexit:
  br label %for.end

for.end:
  ret void
}

attributes #0 = { nounwind }

!0 = !{!"omnipotent char", !1}
!1 = !{!"Simple C/C++ TBAA"}
