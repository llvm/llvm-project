; RUN: llc -O3 -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; XFAIL: *
;
; Generate hardware loop when loop 'latch' block is different
; from the loop 'exiting' block.

; CHECK: loop0(.LBB{{.}}_{{.}}, r{{[0-9]+}})
; CHECK: endloop0

define void @test(i32* nocapture %pFL, i16 signext %nBS, i16* nocapture readonly %pHT) #0 {
entry:
  %0 = load i32, i32* %pFL, align 4
  %1 = tail call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %0, i32 246)
  %2 = tail call i64 @llvm.hexagon.S2.asl.r.p(i64 %1, i32 -13)
  %3 = tail call i32 @llvm.hexagon.A2.sat(i64 %2)
  store i32 %3, i32* %pFL, align 4
  %cmp16 = icmp sgt i16 %nBS, 0
  br i1 %cmp16, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %4 = sext i16 %nBS to i32
  br label %for.body

for.body:
  %5 = phi i32 [ %3, %for.body.lr.ph ], [ %.pre, %for.body.for.body_crit_edge ]
  %arrayidx3.phi = phi i32* [ %pFL, %for.body.lr.ph ], [ %arrayidx3.inc, %for.body.for.body_crit_edge ]
  %arrayidx5.phi = phi i16* [ %pHT, %for.body.lr.ph ], [ %arrayidx5.inc, %for.body.for.body_crit_edge ]
  %i.017.pmt = phi i32 [ 1, %for.body.lr.ph ], [ %phitmp, %for.body.for.body_crit_edge ]
  %6 = load i16, i16* %arrayidx5.phi, align 2
  %conv6 = sext i16 %6 to i32
  %7 = tail call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %5, i32 %conv6)
  %8 = tail call i64 @llvm.hexagon.S2.asl.r.p(i64 %7, i32 -13)
  %9 = tail call i32 @llvm.hexagon.A2.sat(i64 %8)
  store i32 %9, i32* %arrayidx3.phi, align 4
  %exitcond = icmp eq i32 %i.017.pmt, %4
  %arrayidx3.inc = getelementptr i32, i32* %arrayidx3.phi, i32 1
  br i1 %exitcond, label %for.end.loopexit, label %for.body.for.body_crit_edge

for.body.for.body_crit_edge:
  %arrayidx5.inc = getelementptr i16, i16* %arrayidx5.phi, i32 1
  %.pre = load i32, i32* %arrayidx3.inc, align 4
  %phitmp = add i32 %i.017.pmt, 1
  br label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

declare i32 @llvm.hexagon.A2.sat(i64) #1

declare i64 @llvm.hexagon.S2.asl.r.p(i64, i32) #1

declare i64 @llvm.hexagon.M2.dpmpyss.s0(i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "ssp-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
