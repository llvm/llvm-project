; RUN: llc -mtriple=hexagon-unknown-elf -mcpu=hexagonv67t < %s | FileCheck %s

;CHECK-NOT: CONST64

define dso_local void @analyze(ptr nocapture %analysisBuffer0, ptr nocapture %analysisBuffer1, ptr nocapture %subband) local_unnamed_addr {
entry:
  %0 = load i64, ptr undef, align 8
  %1 = tail call i64 @llvm.hexagon.S2.vtrunewh(i64 %0, i64 undef)
  %2 = tail call i64 @llvm.hexagon.S2.vtrunowh(i64 %0, i64 undef)
  %_HEXAGON_V64_internal_union.sroa.3.0.extract.shift = and i64 %1, -4294967296
  %3 = shl i64 %2, 32
  %conv15 = ashr exact i64 %3, 32
  %arrayidx16 = getelementptr inbounds i16, ptr %analysisBuffer0, i32 4
  store i64 %_HEXAGON_V64_internal_union.sroa.3.0.extract.shift, ptr %arrayidx16, align 8
  %arrayidx17 = getelementptr inbounds i16, ptr %analysisBuffer1, i32 4
  store i64 %conv15, ptr %arrayidx17, align 8
  %arrayidx18 = getelementptr inbounds i16, ptr %analysisBuffer1, i32 8
  %4 = load i64, ptr %arrayidx18, align 8
  %5 = tail call i64 @llvm.hexagon.M2.mmachs.s1(i64 undef, i64 29819854865948160, i64 %4)
  store i64 %5, ptr %arrayidx18, align 8
  %arrayidx34 = getelementptr inbounds i16, ptr %analysisBuffer0, i32 40
  %6 = load i64, ptr %arrayidx34, align 8
  %7 = tail call i64 @llvm.hexagon.M2.mmachs.s1(i64 undef, i64 282574488406740992, i64 %6)
  %arrayidx35 = getelementptr inbounds i16, ptr %analysisBuffer0, i32 56
  %8 = load i64, ptr %arrayidx35, align 8
  %9 = tail call i64 @llvm.hexagon.M2.mmacls.s1(i64 undef, i64 undef, i64 %8)
  %10 = tail call i64 @llvm.hexagon.M2.mmachs.s1(i64 %5, i64 282574488406740992, i64 %4)
  %11 = load i64, ptr null, align 8
  %12 = tail call i64 @llvm.hexagon.M2.mmacls.s1(i64 %9, i64 27234903028652032, i64 %11)
  %13 = tail call i64 @llvm.hexagon.M2.mmacls.s1(i64 undef, i64 27234903028652032, i64 %4)
  %14 = tail call i64 @llvm.hexagon.M2.mmachs.s1(i64 %10, i64 7661056, i64 %4)
  %_HEXAGON_V64_internal_union53.sroa.3.0.extract.shift = lshr i64 %12, 32
  %_HEXAGON_V64_internal_union62.sroa.3.0.extract.shift = and i64 %13, -4294967296
  %_HEXAGON_V64_internal_union71.sroa.0.0.insert.insert = or i64 %_HEXAGON_V64_internal_union62.sroa.3.0.extract.shift, %_HEXAGON_V64_internal_union53.sroa.3.0.extract.shift
  %_HEXAGON_V64_internal_union79.sroa.4.0.insert.shift = shl i64 %14, 32
  %_HEXAGON_V64_internal_union79.sroa.0.0.insert.ext = and i64 %7, 4294967295
  %_HEXAGON_V64_internal_union79.sroa.0.0.insert.insert = or i64 %_HEXAGON_V64_internal_union79.sroa.4.0.insert.shift, %_HEXAGON_V64_internal_union79.sroa.0.0.insert.ext
  %15 = tail call i64 @llvm.hexagon.M2.mmpyh.s0(i64 %_HEXAGON_V64_internal_union71.sroa.0.0.insert.insert, i64 undef)
  %16 = tail call i64 @llvm.hexagon.A2.vsubw(i64 undef, i64 %15)
  %17 = tail call i64 @llvm.hexagon.A2.vaddw(i64 undef, i64 undef)
  %18 = tail call i64 @llvm.hexagon.S2.asl.i.vw(i64 %17, i32 2)
  %19 = tail call i64 @llvm.hexagon.M2.mmpyl.s0(i64 0, i64 undef)
  %20 = tail call i64 @llvm.hexagon.S2.asl.i.vw(i64 %19, i32 2)
  %21 = tail call i64 @llvm.hexagon.A2.vsubw(i64 undef, i64 %18)
  %22 = tail call i64 @llvm.hexagon.A2.vaddw(i64 %20, i64 %_HEXAGON_V64_internal_union79.sroa.0.0.insert.insert)
  %23 = tail call i64 @llvm.hexagon.M2.mmpyh.s0(i64 %22, i64 undef)
  %24 = tail call i64 @llvm.hexagon.M2.mmpyl.s0(i64 %21, i64 3998767301)
  %25 = tail call i64 @llvm.hexagon.S2.asl.i.vw(i64 %24, i32 2)
  %26 = tail call i64 @llvm.hexagon.A2.vaddw(i64 undef, i64 %23)
  %27 = tail call i64 @llvm.hexagon.A2.vaddw(i64 0, i64 %25)
  %28 = tail call i64 @llvm.hexagon.A2.vaddw(i64 %16, i64 undef)
  %_HEXAGON_V64_internal_union8.sroa.0.0.insert.ext.i = and i64 %26, 4294967295
  store i64 %_HEXAGON_V64_internal_union8.sroa.0.0.insert.ext.i, ptr %subband, align 8
  %_HEXAGON_V64_internal_union17.sroa.5.0.insert.shift.i = shl i64 %28, 32
  %_HEXAGON_V64_internal_union17.sroa.0.0.insert.ext.i = and i64 %27, 4294967295
  %_HEXAGON_V64_internal_union17.sroa.0.0.insert.insert.i = or i64 %_HEXAGON_V64_internal_union17.sroa.5.0.insert.shift.i, %_HEXAGON_V64_internal_union17.sroa.0.0.insert.ext.i
  %arrayidx31.i = getelementptr inbounds i32, ptr %subband, i32 2
  store i64 %_HEXAGON_V64_internal_union17.sroa.0.0.insert.insert.i, ptr %arrayidx31.i, align 8
  %_HEXAGON_V64_internal_union32.sroa.0.0.insert.ext.i = and i64 %17, 4294967295
  %arrayidx46.i = getelementptr inbounds i32, ptr %subband, i32 4
  store i64 %_HEXAGON_V64_internal_union32.sroa.0.0.insert.ext.i, ptr %arrayidx46.i, align 8
  %arrayidx55.i = getelementptr inbounds i32, ptr %subband, i32 6
  store i64 0, ptr %arrayidx55.i, align 8
  %arrayidx64.i = getelementptr inbounds i32, ptr %subband, i32 8
  store i64 0, ptr %arrayidx64.i, align 8
  %arrayidx73.i = getelementptr inbounds i32, ptr %subband, i32 12
  store i64 0, ptr %arrayidx73.i, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.vtrunewh(i64, i64)
declare i64 @llvm.hexagon.S2.vtrunowh(i64, i64)
declare i64 @llvm.hexagon.M2.mmachs.s1(i64, i64, i64)
declare i64 @llvm.hexagon.M2.mmacls.s1(i64, i64, i64)
declare i64 @llvm.hexagon.M2.mmpyh.s0(i64, i64)
declare i64 @llvm.hexagon.A2.vsubw(i64, i64)
declare i64 @llvm.hexagon.A2.vaddw(i64, i64)
declare i64 @llvm.hexagon.S2.asl.i.vw(i64, i32)
declare i64 @llvm.hexagon.M2.mmpyl.s0(i64, i64)
