; RUN: llc -mtriple=hexagon-unknown-elf -mcpu=hexagonv67t < %s | FileCheck %s

;CHECK-NOT: CONST64

define dso_local void @analyze(i16* nocapture %analysisBuffer0, i16* nocapture %analysisBuffer1, i32* nocapture %subband) local_unnamed_addr {
entry:
  %0 = load i64, i64* undef, align 8
  %1 = tail call i64 @llvm.hexagon.S2.vtrunewh(i64 %0, i64 undef)
  %2 = tail call i64 @llvm.hexagon.S2.vtrunowh(i64 %0, i64 undef)
  %_HEXAGON_V64_internal_union.sroa.3.0.extract.shift = and i64 %1, -4294967296
  %3 = shl i64 %2, 32
  %conv15 = ashr exact i64 %3, 32
  %arrayidx16 = getelementptr inbounds i16, i16* %analysisBuffer0, i32 4
  %4 = bitcast i16* %arrayidx16 to i64*
  store i64 %_HEXAGON_V64_internal_union.sroa.3.0.extract.shift, i64* %4, align 8
  %arrayidx17 = getelementptr inbounds i16, i16* %analysisBuffer1, i32 4
  %5 = bitcast i16* %arrayidx17 to i64*
  store i64 %conv15, i64* %5, align 8
  %arrayidx18 = getelementptr inbounds i16, i16* %analysisBuffer1, i32 8
  %6 = bitcast i16* %arrayidx18 to i64*
  %7 = load i64, i64* %6, align 8
  %8 = tail call i64 @llvm.hexagon.M2.mmachs.s1(i64 undef, i64 29819854865948160, i64 %7)
  store i64 %8, i64* %6, align 8
  %arrayidx34 = getelementptr inbounds i16, i16* %analysisBuffer0, i32 40
  %9 = bitcast i16* %arrayidx34 to i64*
  %10 = load i64, i64* %9, align 8
  %11 = tail call i64 @llvm.hexagon.M2.mmachs.s1(i64 undef, i64 282574488406740992, i64 %10)
  %arrayidx35 = getelementptr inbounds i16, i16* %analysisBuffer0, i32 56
  %12 = bitcast i16* %arrayidx35 to i64*
  %13 = load i64, i64* %12, align 8
  %14 = tail call i64 @llvm.hexagon.M2.mmacls.s1(i64 undef, i64 undef, i64 %13)
  %15 = tail call i64 @llvm.hexagon.M2.mmachs.s1(i64 %8, i64 282574488406740992, i64 %7)
  %16 = load i64, i64* null, align 8
  %17 = tail call i64 @llvm.hexagon.M2.mmacls.s1(i64 %14, i64 27234903028652032, i64 %16)
  %18 = tail call i64 @llvm.hexagon.M2.mmacls.s1(i64 undef, i64 27234903028652032, i64 %7)
  %19 = tail call i64 @llvm.hexagon.M2.mmachs.s1(i64 %15, i64 7661056, i64 %7)
  %_HEXAGON_V64_internal_union53.sroa.3.0.extract.shift = lshr i64 %17, 32
  %_HEXAGON_V64_internal_union62.sroa.3.0.extract.shift = and i64 %18, -4294967296
  %_HEXAGON_V64_internal_union71.sroa.0.0.insert.insert = or i64 %_HEXAGON_V64_internal_union62.sroa.3.0.extract.shift, %_HEXAGON_V64_internal_union53.sroa.3.0.extract.shift
  %_HEXAGON_V64_internal_union79.sroa.4.0.insert.shift = shl i64 %19, 32
  %_HEXAGON_V64_internal_union79.sroa.0.0.insert.ext = and i64 %11, 4294967295
  %_HEXAGON_V64_internal_union79.sroa.0.0.insert.insert = or i64 %_HEXAGON_V64_internal_union79.sroa.4.0.insert.shift, %_HEXAGON_V64_internal_union79.sroa.0.0.insert.ext
  %20 = bitcast i32* %subband to i64*
  %21 = tail call i64 @llvm.hexagon.M2.mmpyh.s0(i64 %_HEXAGON_V64_internal_union71.sroa.0.0.insert.insert, i64 undef)
  %22 = tail call i64 @llvm.hexagon.A2.vsubw(i64 undef, i64 %21)
  %23 = tail call i64 @llvm.hexagon.A2.vaddw(i64 undef, i64 undef)
  %24 = tail call i64 @llvm.hexagon.S2.asl.i.vw(i64 %23, i32 2)
  %25 = tail call i64 @llvm.hexagon.M2.mmpyl.s0(i64 0, i64 undef)
  %26 = tail call i64 @llvm.hexagon.S2.asl.i.vw(i64 %25, i32 2)
  %27 = tail call i64 @llvm.hexagon.A2.vsubw(i64 undef, i64 %24)
  %28 = tail call i64 @llvm.hexagon.A2.vaddw(i64 %26, i64 %_HEXAGON_V64_internal_union79.sroa.0.0.insert.insert)
  %29 = tail call i64 @llvm.hexagon.M2.mmpyh.s0(i64 %28, i64 undef)
  %30 = tail call i64 @llvm.hexagon.M2.mmpyl.s0(i64 %27, i64 3998767301)
  %31 = tail call i64 @llvm.hexagon.S2.asl.i.vw(i64 %30, i32 2)
  %32 = tail call i64 @llvm.hexagon.A2.vaddw(i64 undef, i64 %29)
  %33 = tail call i64 @llvm.hexagon.A2.vaddw(i64 0, i64 %31)
  %34 = tail call i64 @llvm.hexagon.A2.vaddw(i64 %22, i64 undef)
  %_HEXAGON_V64_internal_union8.sroa.0.0.insert.ext.i = and i64 %32, 4294967295
  store i64 %_HEXAGON_V64_internal_union8.sroa.0.0.insert.ext.i, i64* %20, align 8
  %_HEXAGON_V64_internal_union17.sroa.5.0.insert.shift.i = shl i64 %34, 32
  %_HEXAGON_V64_internal_union17.sroa.0.0.insert.ext.i = and i64 %33, 4294967295
  %_HEXAGON_V64_internal_union17.sroa.0.0.insert.insert.i = or i64 %_HEXAGON_V64_internal_union17.sroa.5.0.insert.shift.i, %_HEXAGON_V64_internal_union17.sroa.0.0.insert.ext.i
  %arrayidx31.i = getelementptr inbounds i32, i32* %subband, i32 2
  %35 = bitcast i32* %arrayidx31.i to i64*
  store i64 %_HEXAGON_V64_internal_union17.sroa.0.0.insert.insert.i, i64* %35, align 8
  %_HEXAGON_V64_internal_union32.sroa.0.0.insert.ext.i = and i64 %23, 4294967295
  %arrayidx46.i = getelementptr inbounds i32, i32* %subband, i32 4
  %36 = bitcast i32* %arrayidx46.i to i64*
  store i64 %_HEXAGON_V64_internal_union32.sroa.0.0.insert.ext.i, i64* %36, align 8
  %arrayidx55.i = getelementptr inbounds i32, i32* %subband, i32 6
  %37 = bitcast i32* %arrayidx55.i to i64*
  store i64 0, i64* %37, align 8
  %arrayidx64.i = getelementptr inbounds i32, i32* %subband, i32 8
  %38 = bitcast i32* %arrayidx64.i to i64*
  store i64 0, i64* %38, align 8
  %arrayidx73.i = getelementptr inbounds i32, i32* %subband, i32 12
  %39 = bitcast i32* %arrayidx73.i to i64*
  store i64 0, i64* %39, align 8
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
