; RUN: opt -mtriple=hexagon-- -mcpu=hexagonv68 -mattr=+hvxv68,+hvx-length128b \
; RUN:   -passes='loop(hexagon-vlcr)' -S \
; RUN:   -pass-remarks=hexagon-vlcr -pass-remarks-missed=hexagon-vlcr \
; RUN:   %s -o /dev/null 2>&1 | FileCheck %s

;; Test that HexagonVectorLoopCarriedReuse emits optimization remarks.

;; -- Success: reused loop-carried vector value --
; CHECK: remark: {{.*}} reused loop-carried vector value

@W = external local_unnamed_addr global i32, align 4

define void @test_reuse(ptr noalias nocapture readonly %src, ptr noalias nocapture %dst, i32 %stride) {
entry:
  %add.ptr = getelementptr inbounds i8, ptr %src, i32 %stride
  %0 = load i32, ptr @W, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %for.body.preheader, label %for.end

for.body.preheader:
  %1 = load <32 x i32>, ptr %add.ptr, align 128
  %incdec.ptr1 = getelementptr inbounds i8, ptr %add.ptr, i32 128
  %2 = load <32 x i32>, ptr %src, align 128
  %incdec.ptr = getelementptr inbounds i8, ptr %src, i32 128
  br label %for.body

for.body:
  %out.phi = phi ptr [ %dst, %for.body.preheader ], [ %out.inc, %for.body ]
  %p1.phi = phi ptr [ %incdec.ptr1, %for.body.preheader ], [ %p1.inc, %for.body ]
  %p0.phi = phi ptr [ %incdec.ptr, %for.body.preheader ], [ %p0.inc, %for.body ]
  %i.phi = phi i32 [ 0, %for.body.preheader ], [ %i.inc, %for.body ]
  %a.phi = phi <32 x i32> [ %2, %for.body.preheader ], [ %3, %for.body ]
  %b.phi = phi <32 x i32> [ %1, %for.body.preheader ], [ %4, %for.body ]
  %p0.inc = getelementptr inbounds <32 x i32>, ptr %p0.phi, i32 1
  %3 = load <32 x i32>, ptr %p0.phi, align 128
  %p1.inc = getelementptr inbounds <32 x i32>, ptr %p1.phi, i32 1
  %4 = load <32 x i32>, ptr %p1.phi, align 128
  %max1 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %a.phi, <32 x i32> %b.phi)
  %max2 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %3, <32 x i32> %4)
  %sum = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %max1, <32 x i32> %max2)
  store <32 x i32> %sum, ptr %out.phi, align 128
  %out.inc = getelementptr inbounds <32 x i32>, ptr %out.phi, i32 1
  %i.inc = add nuw nsw i32 %i.phi, 1
  %exitcond = icmp slt i32 %i.inc, %0
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

;; -- Missed: loop has multiple basic blocks --
; CHECK: remark: {{.*}} loop has multiple basic blocks

define void @test_multi_bb(ptr noalias nocapture readonly %src, ptr noalias nocapture %dst, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.latch ]
  %p = phi ptr [ %src, %entry ], [ %p.inc, %for.latch ]
  %q = phi ptr [ %dst, %entry ], [ %q.inc, %for.latch ]
  %val = load <32 x i32>, ptr %p, align 128
  %check = icmp eq i32 %i, 0
  br i1 %check, label %if.then, label %for.latch

if.then:
  store <32 x i32> %val, ptr %q, align 128
  br label %for.latch

for.latch:
  %p.inc = getelementptr inbounds <32 x i32>, ptr %p, i32 1
  %q.inc = getelementptr inbounds <32 x i32>, ptr %q, i32 1
  %i.inc = add nuw nsw i32 %i, 1
  %exitcond = icmp ne i32 %i.inc, %n
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

declare <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32>, <32 x i32>)
