; RUN: llc -mtriple=hexagon < %s | FileCheck %s
;
; Check that we no longer generate 4 inserts.
; CHECK: combine(r{{[0-9]+}}.l,r{{[0-9]+}}.l)
; CHECK: combine(r{{[0-9]+}}.l,r{{[0-9]+}}.l)
; CHECK-NOT: insert

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

%struct.a = type { i16 }

define i32 @fun(ptr nocapture %pData, i64 %c, ptr nocapture %d, ptr nocapture %e, ptr nocapture %f) #0 {
entry:
  %0 = load i16, ptr %pData, align 2, !tbaa !0
  %conv185 = sext i16 %0 to i32
  %shr86 = ashr i32 %conv185, 2
  %cmp87 = icmp sgt i32 %shr86, 0
  br i1 %cmp87, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %h.sroa.0.0.extract.trunc = trunc i64 %c to i32
  %sext = shl i32 %h.sroa.0.0.extract.trunc, 16
  %conv8 = ashr exact i32 %sext, 16
  %l.sroa.2.4.extract.shift = lshr i64 %c, 32
  %sext76 = ashr i32 %h.sroa.0.0.extract.trunc, 16
  %m.sroa.2.6.extract.shift = lshr i64 %c, 48
  %sext7980 = shl nuw nsw i64 %l.sroa.2.4.extract.shift, 16
  %sext79 = trunc i64 %sext7980 to i32
  %conv38 = ashr exact i32 %sext79, 16
  %sext8283 = shl nuw nsw i64 %m.sroa.2.6.extract.shift, 16
  %sext82 = trunc i64 %sext8283 to i32
  %conv53 = ashr exact i32 %sext82, 16
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %arrayidx.phi = phi ptr [ %d, %for.body.lr.ph ], [ %arrayidx.inc, %for.body ]
  %arrayidx30.phi = phi ptr [ %f, %for.body.lr.ph ], [ %arrayidx30.inc, %for.body ]
  %arrayidx60.phi = phi ptr [ %e, %for.body.lr.ph ], [ %arrayidx60.inc, %for.body ]
  %j.088.pmt = phi i32 [ 0, %for.body.lr.ph ], [ %inc.pmt, %for.body ]
  %1 = load i64, ptr %arrayidx.phi, align 8, !tbaa !1
  %n_union3.sroa.0.0.extract.trunc = trunc i64 %1 to i32
  %n_union3.sroa.1.4.extract.shift = lshr i64 %1, 32
  %2 = tail call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %n_union3.sroa.0.0.extract.trunc, i32 %conv8)
  %3 = tail call i64 @llvm.hexagon.S2.asl.r.p(i64 %2, i32 -25)
  %conv9 = trunc i64 %3 to i32
  %4 = tail call i32 @llvm.hexagon.A2.sath(i32 %conv9)
  %n_union13.sroa.1.4.extract.trunc = trunc i64 %n_union3.sroa.1.4.extract.shift to i32
  %5 = tail call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %n_union13.sroa.1.4.extract.trunc, i32 %sext76)
  %6 = tail call i64 @llvm.hexagon.S2.asl.r.p(i64 %5, i32 -25)
  %conv24 = trunc i64 %6 to i32
  %7 = tail call i32 @llvm.hexagon.A2.sath(i32 %conv24)
  %8 = load i64, ptr %arrayidx30.phi, align 8, !tbaa !1
  %n_union28.sroa.0.0.extract.trunc = trunc i64 %8 to i32
  %n_union28.sroa.1.4.extract.shift = lshr i64 %8, 32
  %9 = tail call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %n_union28.sroa.0.0.extract.trunc, i32 %conv38)
  %10 = tail call i64 @llvm.hexagon.S2.asl.r.p(i64 %9, i32 -25)
  %conv39 = trunc i64 %10 to i32
  %11 = tail call i32 @llvm.hexagon.A2.sath(i32 %conv39)
  %n_union43.sroa.1.4.extract.trunc = trunc i64 %n_union28.sroa.1.4.extract.shift to i32
  %12 = tail call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %n_union43.sroa.1.4.extract.trunc, i32 %conv53)
  %13 = tail call i64 @llvm.hexagon.S2.asl.r.p(i64 %12, i32 -25)
  %conv54 = trunc i64 %13 to i32
  %14 = tail call i32 @llvm.hexagon.A2.sath(i32 %conv54)
  %n_union.sroa.3.6.insert.ext = zext i32 %14 to i64
  %n_union.sroa.3.6.insert.shift = shl i64 %n_union.sroa.3.6.insert.ext, 48
  %conv40.mask = and i32 %11, 65535
  %n_union.sroa.2.4.insert.ext = zext i32 %conv40.mask to i64
  %n_union.sroa.2.4.insert.shift = shl nuw nsw i64 %n_union.sroa.2.4.insert.ext, 32
  %conv25.mask = and i32 %7, 65535
  %n_union.sroa.1.2.insert.ext = zext i32 %conv25.mask to i64
  %n_union.sroa.1.2.insert.shift = shl nuw nsw i64 %n_union.sroa.1.2.insert.ext, 16
  %conv10.mask = and i32 %4, 65535
  %n_union.sroa.0.0.insert.ext = zext i32 %conv10.mask to i64
  %n_union.sroa.2.4.insert.insert = or i64 %n_union.sroa.1.2.insert.shift, %n_union.sroa.0.0.insert.ext
  %n_union.sroa.1.2.insert.insert = or i64 %n_union.sroa.2.4.insert.insert, %n_union.sroa.2.4.insert.shift
  %n_union.sroa.0.0.insert.insert = or i64 %n_union.sroa.1.2.insert.insert, %n_union.sroa.3.6.insert.shift
  %15 = load i64, ptr %arrayidx60.phi, align 8, !tbaa !1
  %16 = tail call i64 @llvm.hexagon.A2.vaddhs(i64 %15, i64 %n_union.sroa.0.0.insert.insert)
  store i64 %16, ptr %arrayidx60.phi, align 8, !tbaa !1
  %inc.pmt = add i32 %j.088.pmt, 1
  %17 = load i16, ptr %pData, align 2, !tbaa !0
  %conv1 = sext i16 %17 to i32
  %shr = ashr i32 %conv1, 2
  %cmp = icmp slt i32 %inc.pmt, %shr
  %arrayidx.inc = getelementptr i64, ptr %arrayidx.phi, i32 1
  %arrayidx30.inc = getelementptr i64, ptr %arrayidx30.phi, i32 1
  %arrayidx60.inc = getelementptr i64, ptr %arrayidx60.phi, i32 1
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret i32 0
}

declare i32 @llvm.hexagon.A2.sath(i32) #1

declare i64 @llvm.hexagon.S2.asl.r.p(i64, i32) #1

declare i64 @llvm.hexagon.M2.dpmpyss.s0(i32, i32) #1

declare i64 @llvm.hexagon.A2.vaddhs(i64, i64) #1

attributes #0 = { nounwind "fp-contract-model"="standard" "frame-pointer"="non-leaf" "realign-stack" "relocation-model"="static" "ssp-buffers-size"="8" }
attributes #1 = { nounwind readnone }

!0 = !{!"short", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
