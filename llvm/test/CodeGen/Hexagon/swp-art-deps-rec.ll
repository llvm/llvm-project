; REQUIRES: asserts

; RUN: llc -march=hexagon -mcpu=hexagonv65 -O3 -debug-only=pipeliner \
; RUN: < %s 2>&1 -pipeliner-experimental-cg=true | FileCheck %s

; Test that the artificial dependences are ignored while computing the
; circuits.

; The recurrence should be 1 here. If we do not ignore artificial deps,
; it will be greater.
; CHECK: rec=1,

define void @foo(i32 %size) #0 {
entry:
  %add = add nsw i32 0, 4
  %shr = ashr i32 %size, 1
  br i1 undef, label %L57.us, label %L57.us.ur

L57.us:
  %R9.0470.us = phi i32 [ %sub40.us.3, %L57.us ], [ undef, %entry ]
  %sub40.us.3 = add i32 %R9.0470.us, -64
  br i1 undef, label %L57.us, label %for.cond22.for.end_crit_edge.us.ur-lcssa

for.cond22.for.end_crit_edge.us.ur-lcssa:
  %inc.us.3.lcssa = phi i32 [ undef, %L57.us ]
  %sub40.us.3.lcssa = phi i32 [ %sub40.us.3, %L57.us ]
  %0 = icmp eq i32 %inc.us.3.lcssa, %shr
  br i1 %0, label %for.cond22.for.end_crit_edge.us, label %L57.us.ur

L57.us.ur:
  %R15_14.0478.us.ur = phi i64 [ %1, %L57.us.ur ], [ 0, %entry ], [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ]
  %R13_12.0477.us.ur = phi i64 [ %14, %L57.us.ur ], [ 0, %entry ], [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ]
  %R11_10.0476.us.ur = phi i64 [ %8, %L57.us.ur ], [ 0, %entry ], [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ]
  %R7_6.0475.us.ur = phi i64 [ %7, %L57.us.ur ], [ 0, %entry ], [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ]
  %R5_4.2474.us.ur = phi i64 [ %16, %L57.us.ur ], [ undef, %entry ], [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ]
  %R3_2.0473.us.ur = phi i64 [ %9, %L57.us.ur ], [ 0, %entry ], [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ]
  %R1_0.0472.us.ur = phi i64 [ %15, %L57.us.ur ], [ undef, %entry ], [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ]
  %kk.0471.us.ur = phi i32 [ %inc.us.ur, %L57.us.ur ], [ 0, %entry ], [ %inc.us.3.lcssa, %for.cond22.for.end_crit_edge.us.ur-lcssa ]
  %R9.0470.us.ur = phi i32 [ %sub40.us.ur, %L57.us.ur ], [ undef, %entry ], [ %sub40.us.3.lcssa, %for.cond22.for.end_crit_edge.us.ur-lcssa ]
  %R8.0469.us.ur = phi i32 [ %sub34.us.ur, %L57.us.ur ], [ undef, %entry ], [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ]
  %1 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %R15_14.0478.us.ur, i64 %R1_0.0472.us.ur, i64 %R3_2.0473.us.ur)
  %2 = tail call i64 @llvm.hexagon.S2.shuffeh(i64 %R5_4.2474.us.ur, i64 %R7_6.0475.us.ur)
  %3 = inttoptr i32 %R9.0470.us.ur to i16*
  %4 = load i16, i16* %3, align 2
  %conv27.us.ur = sext i16 %4 to i32
  %sub28.us.ur = add i32 %R9.0470.us.ur, -8
  %5 = inttoptr i32 %R8.0469.us.ur to i16*
  %6 = load i16, i16* %5, align 2
  %conv30.us.ur = sext i16 %6 to i32
  %sub31.us.ur = add i32 %R8.0469.us.ur, -8
  %7 = tail call i64 @llvm.hexagon.A2.combinew(i32 %conv27.us.ur, i32 %conv30.us.ur)
  %8 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %R11_10.0476.us.ur, i64 %R1_0.0472.us.ur, i64 %2)
  %9 = tail call i64 @llvm.hexagon.S2.shuffeh(i64 %7, i64 %R5_4.2474.us.ur)
  %10 = inttoptr i32 %sub31.us.ur to i16*
  %11 = load i16, i16* %10, align 2
  %conv33.us.ur = sext i16 %11 to i32
  %sub34.us.ur = add i32 %R8.0469.us.ur, -16
  %conv35.us.ur = trunc i64 %9 to i32
  %12 = inttoptr i32 %sub28.us.ur to i16*
  %13 = load i16, i16* %12, align 2
  %conv39.us.ur = sext i16 %13 to i32
  %sub40.us.ur = add i32 %R9.0470.us.ur, -16
  %14 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %R13_12.0477.us.ur, i64 %R1_0.0472.us.ur, i64 %9)
  %15 = tail call i64 @llvm.hexagon.A2.combinew(i32 %conv35.us.ur, i32 undef)
  %16 = tail call i64 @llvm.hexagon.A2.combinew(i32 %conv39.us.ur, i32 %conv33.us.ur)
  %inc.us.ur = add nsw i32 %kk.0471.us.ur, 1
  %exitcond535.ur = icmp eq i32 %inc.us.ur, %shr
  br i1 %exitcond535.ur, label %for.cond22.for.end_crit_edge.us.ur-lcssa572, label %L57.us.ur

for.cond22.for.end_crit_edge.us.ur-lcssa572:
  %.lcssa730 = phi i64 [ %14, %L57.us.ur ]
  %.lcssa729 = phi i64 [ %8, %L57.us.ur ]
  %.lcssa728 = phi i64 [ %1, %L57.us.ur ]
  %extract.t652 = trunc i64 %.lcssa730 to i32
  %extract661 = lshr i64 %.lcssa729, 32
  %extract.t662 = trunc i64 %extract661 to i32
  %extract.t664 = trunc i64 %.lcssa728 to i32
  br label %for.cond22.for.end_crit_edge.us

for.cond22.for.end_crit_edge.us:
  %.lcssa551.off0 = phi i32 [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ], [ %extract.t652, %for.cond22.for.end_crit_edge.us.ur-lcssa572 ]
  %.lcssa550.off32 = phi i32 [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ], [ %extract.t662, %for.cond22.for.end_crit_edge.us.ur-lcssa572 ]
  %.lcssa549.off0 = phi i32 [ undef, %for.cond22.for.end_crit_edge.us.ur-lcssa ], [ %extract.t664, %for.cond22.for.end_crit_edge.us.ur-lcssa572 ]
  %17 = inttoptr i32 %add to i32*
  store i32 %.lcssa549.off0, i32* %17, align 4
  %add.ptr61.us = getelementptr inbounds i8, i8* null, i32 32
  %18 = bitcast i8* %add.ptr61.us to i32*
  store i32 %.lcssa551.off0, i32* %18, align 4
  %19 = bitcast i8* undef to i32*
  store i32 %.lcssa550.off32, i32* %19, align 4
  call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vdmacs.s0(i64, i64, i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.shuffeh(i64, i64) #1

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { noreturn nounwind }
