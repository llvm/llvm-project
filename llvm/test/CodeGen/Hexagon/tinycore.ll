; RUN: llc -march=hexagon -mcpu=hexagonv67t < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv65 < %s | FileCheck --check-prefix=CHECK-BIG %s

; Test that the tiny core architecture generates 3 slot packets at most and
; a single load/store per packet at most.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: {
; CHECK-NEXT: mpy
; CHECK-NEXT: combine
; CHECK-NEXT: memw
; CHECK-NEXT: }
; CHECK: memw
; CHECK: } :endloop0

; Test the loop contains a single packet with 4 instructions.
; CHECK-BIG:  loop0(.LBB0_[[LOOP:.]],
; CHECK-BIG: .LBB0_[[LOOP]]:
; CHECK-BIG: {
; CHECK-BIG: += mpyi
; CHECK-BIG-NEXT: = combine
; CHECK-BIG-NEXT: = memw
; CHECK-BIG-NEXT: = memw
; CHECK-BIG-NEXT: } :endloop0

define i32 @test(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i32 %n) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body, label %for.end

for.body:
  %sum.010 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx.phi = phi ptr [ %arrayidx.inc, %for.body ], [ %a, %entry ]
  %arrayidx1.phi = phi ptr [ %arrayidx1.inc, %for.body ], [ %b, %entry ]
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %0 = load i32, ptr %arrayidx.phi, align 4
  %1 = load i32, ptr %arrayidx1.phi, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, %sum.010
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %n
  %arrayidx.inc = getelementptr i32, ptr %arrayidx.phi, i32 1
  %arrayidx1.inc = getelementptr i32, ptr %arrayidx1.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa
}

