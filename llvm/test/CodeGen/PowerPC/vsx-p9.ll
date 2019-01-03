; RUN: llc -verify-machineinstrs -mcpu=pwr9 \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 \
; RUN:   -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s

@uca = global <16 x i8> zeroinitializer, align 16
@ucb = global <16 x i8> zeroinitializer, align 16
@sca = global <16 x i8> zeroinitializer, align 16
@scb = global <16 x i8> zeroinitializer, align 16
@usa = global <8 x i16> zeroinitializer, align 16
@usb = global <8 x i16> zeroinitializer, align 16
@ssa = global <8 x i16> zeroinitializer, align 16
@ssb = global <8 x i16> zeroinitializer, align 16
@uia = global <4 x i32> zeroinitializer, align 16
@uib = global <4 x i32> zeroinitializer, align 16
@sia = global <4 x i32> zeroinitializer, align 16
@sib = global <4 x i32> zeroinitializer, align 16
@ulla = global <2 x i64> zeroinitializer, align 16
@ullb = global <2 x i64> zeroinitializer, align 16
@slla = global <2 x i64> zeroinitializer, align 16
@sllb = global <2 x i64> zeroinitializer, align 16
@uxa = global <1 x i128> zeroinitializer, align 16
@uxb = global <1 x i128> zeroinitializer, align 16
@sxa = global <1 x i128> zeroinitializer, align 16
@sxb = global <1 x i128> zeroinitializer, align 16
@vfa = global <4 x float> zeroinitializer, align 16
@vfb = global <4 x float> zeroinitializer, align 16
@vda = global <2 x double> zeroinitializer, align 16
@vdb = global <2 x double> zeroinitializer, align 16

define void @_Z4testv() {
entry:
; CHECK-LABEL: @_Z4testv
  %0 = load <16 x i8>, <16 x i8>* @uca, align 16
  %1 = load <16 x i8>, <16 x i8>* @ucb, align 16
  %add.i = add <16 x i8> %1, %0
  tail call void (...) @sink(<16 x i8> %add.i)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 3 
; CHECK: vaddubm 2, 3, 2
; CHECK: stxv 34,
; CHECK: bl sink
  %2 = load <16 x i8>, <16 x i8>* @sca, align 16
  %3 = load <16 x i8>, <16 x i8>* @scb, align 16
  %add.i22 = add <16 x i8> %3, %2
  tail call void (...) @sink(<16 x i8> %add.i22)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 3 
; CHECK: vaddubm 2, 3, 2
; CHECK: stxv 34,
; CHECK: bl sink
  %4 = load <8 x i16>, <8 x i16>* @usa, align 16
  %5 = load <8 x i16>, <8 x i16>* @usb, align 16
  %add.i21 = add <8 x i16> %5, %4
  tail call void (...) @sink(<8 x i16> %add.i21)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 3 
; CHECK: vadduhm 2, 3, 2
; CHECK: stxv 34,
; CHECK: bl sink
  %6 = load <8 x i16>, <8 x i16>* @ssa, align 16
  %7 = load <8 x i16>, <8 x i16>* @ssb, align 16
  %add.i20 = add <8 x i16> %7, %6
  tail call void (...) @sink(<8 x i16> %add.i20)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 3 
; CHECK: vadduhm 2, 3, 2
; CHECK: stxv 34,
; CHECK: bl sink
  %8 = load <4 x i32>, <4 x i32>* @uia, align 16
  %9 = load <4 x i32>, <4 x i32>* @uib, align 16
  %add.i19 = add <4 x i32> %9, %8
  tail call void (...) @sink(<4 x i32> %add.i19)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 3 
; CHECK: vadduwm 2, 3, 2
; CHECK: stxv 34,
; CHECK: bl sink
  %10 = load <4 x i32>, <4 x i32>* @sia, align 16
  %11 = load <4 x i32>, <4 x i32>* @sib, align 16
  %add.i18 = add <4 x i32> %11, %10
  tail call void (...) @sink(<4 x i32> %add.i18)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 3 
; CHECK: vadduwm 2, 3, 2
; CHECK: stxv 34,
; CHECK: bl sink
  %12 = load <2 x i64>, <2 x i64>* @ulla, align 16
  %13 = load <2 x i64>, <2 x i64>* @ullb, align 16
  %add.i17 = add <2 x i64> %13, %12
  tail call void (...) @sink(<2 x i64> %add.i17)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 3 
; CHECK: vaddudm 2, 3, 2
; CHECK: stxv 34,
; CHECK: bl sink
  %14 = load <2 x i64>, <2 x i64>* @slla, align 16
  %15 = load <2 x i64>, <2 x i64>* @sllb, align 16
  %add.i16 = add <2 x i64> %15, %14
  tail call void (...) @sink(<2 x i64> %add.i16)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 3 
; CHECK: vaddudm 2, 3, 2
; CHECK: stxv 34,
; CHECK: bl sink
  %16 = load <1 x i128>, <1 x i128>* @uxa, align 16
  %17 = load <1 x i128>, <1 x i128>* @uxb, align 16
  %add.i15 = add <1 x i128> %17, %16
  tail call void (...) @sink(<1 x i128> %add.i15)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 3 
; CHECK: vadduqm 2, 3, 2
; CHECK: stxv 34,
; CHECK: bl sink
  %18 = load <1 x i128>, <1 x i128>* @sxa, align 16
  %19 = load <1 x i128>, <1 x i128>* @sxb, align 16
  %add.i14 = add <1 x i128> %19, %18
  tail call void (...) @sink(<1 x i128> %add.i14)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 3 
; CHECK: vadduqm 2, 3, 2
; CHECK: stxv 34,
; CHECK: bl sink
  %20 = load <4 x float>, <4 x float>* @vfa, align 16
  %21 = load <4 x float>, <4 x float>* @vfb, align 16
  %add.i13 = fadd <4 x float> %20, %21
  tail call void (...) @sink(<4 x float> %add.i13)
; CHECK: lxvx 0, 0, 3
; CHECK: lxvx 1, 0, 3 
; CHECK: xvaddsp 34, 0, 1
; CHECK: stxv 34,
; CHECK: bl sink
  %22 = load <2 x double>, <2 x double>* @vda, align 16
  %23 = load <2 x double>, <2 x double>* @vdb, align 16
  %add.i12 = fadd <2 x double> %22, %23
  tail call void (...) @sink(<2 x double> %add.i12)
; CHECK: lxvx 0, 0, 3
; CHECK: lxvx 1, 0, 3 
; CHECK: xvadddp 0, 0, 1
; CHECK: stxv 0,
; CHECK: bl sink
  ret void
}

; Function Attrs: nounwind readnone
define <4 x float> @testXVIEXPSP(<4 x i32> %a, <4 x i32> %b) {
entry:
  %0 = tail call <4 x float> @llvm.ppc.vsx.xviexpsp(<4 x i32> %a, <4 x i32> %b)
  ret <4 x float> %0
; CHECK-LABEL: testXVIEXPSP
; CHECK: xviexpsp 34, 34, 35
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <4 x float> @llvm.ppc.vsx.xviexpsp(<4 x i32>, <4 x i32>)

; Function Attrs: nounwind readnone
define <2 x double> @testXVIEXPDP(<2 x i64> %a, <2 x i64> %b) {
entry:
  %0 = tail call <2 x double> @llvm.ppc.vsx.xviexpdp(<2 x i64> %a, <2 x i64> %b)
  ret <2 x double> %0
; CHECK-LABEL: testXVIEXPDP
; CHECK: xviexpdp 34, 34, 35
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <2 x double> @llvm.ppc.vsx.xviexpdp(<2 x i64>, <2 x i64>)

define <16 x i8> @testVSLV(<16 x i8> %a, <16 x i8> %b) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.altivec.vslv(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %0
; CHECK-LABEL: testVSLV
; CHECK: vslv 2, 2, 3
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.ppc.altivec.vslv(<16 x i8>, <16 x i8>)

; Function Attrs: nounwind readnone
define <16 x i8> @testVSRV(<16 x i8> %a, <16 x i8> %b) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.altivec.vsrv(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %0
; CHECK-LABEL: testVSRV
; CHECK: vsrv 2, 2, 3
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.ppc.altivec.vsrv(<16 x i8>, <16 x i8>)

; Function Attrs: nounwind readnone
define <8 x i16> @testXVCVSPHP(<4 x float> %a) {
entry:
; CHECK-LABEL: testXVCVSPHP
; CHECK: xvcvsphp 34, 34
; CHECK: blr
  %0 = tail call <4 x float> @llvm.ppc.vsx.xvcvsphp(<4 x float> %a)
  %1 = bitcast <4 x float> %0 to <8 x i16>
  ret <8 x i16> %1
}

; Function Attrs: nounwind readnone
define <4 x i32> @testVRLWMI(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
entry:
; CHECK-LABEL: testVRLWMI
; CHECK: vrlwmi 3, 2, 4
; CHECK: blr
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vrlwmi(<4 x i32> %a, <4 x i32> %c, <4 x i32> %b)
  ret <4 x i32> %0
}

; Function Attrs: nounwind readnone
define <2 x i64> @testVRLDMI(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) {
entry:
; CHECK-LABEL: testVRLDMI
; CHECK: vrldmi 3, 2, 4
; CHECK: blr
  %0 = tail call <2 x i64> @llvm.ppc.altivec.vrldmi(<2 x i64> %a, <2 x i64> %c, <2 x i64> %b)
  ret <2 x i64> %0
}

; Function Attrs: nounwind readnone
define <4 x i32> @testVRLWNM(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
entry:
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vrlwnm(<4 x i32> %a, <4 x i32> %b)
  %and.i = and <4 x i32> %0, %c
  ret <4 x i32> %and.i
; CHECK-LABEL: testVRLWNM
; CHECK: vrlwnm 2, 2, 3
; CHECK: xxland 34, 34, 36
; CHECK: blr
}

; Function Attrs: nounwind readnone
define <2 x i64> @testVRLDNM(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) {
entry:
  %0 = tail call <2 x i64> @llvm.ppc.altivec.vrldnm(<2 x i64> %a, <2 x i64> %b)
  %and.i = and <2 x i64> %0, %c
  ret <2 x i64> %and.i
; CHECK-LABEL: testVRLDNM
; CHECK: vrldnm 2, 2, 3
; CHECK: xxland 34, 34, 36
; CHECK: blr
}

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.ppc.vsx.xvcvsphp(<4 x float>)

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.altivec.vrlwmi(<4 x i32>, <4 x i32>, <4 x i32>)

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.vrldmi(<2 x i64>, <2 x i64>, <2 x i64>)

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.altivec.vrlwnm(<4 x i32>, <4 x i32>)

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.vrldnm(<2 x i64>, <2 x i64>)

define <4 x i32> @testXVXEXPSP(<4 x float> %a) {
entry:
  %0 = tail call <4 x i32> @llvm.ppc.vsx.xvxexpsp(<4 x float> %a)
  ret <4 x i32> %0
; CHECK-LABEL: testXVXEXPSP
; CHECK: xvxexpsp 34, 34
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.vsx.xvxexpsp(<4 x float>)

; Function Attrs: nounwind readnone
define <2 x i64> @testXVXEXPDP(<2 x double> %a) {
entry:
  %0 = tail call <2 x i64> @llvm.ppc.vsx.xvxexpdp(<2 x double> %a)
  ret <2 x i64> %0
; CHECK-LABEL: testXVXEXPDP
; CHECK: xvxexpdp 34, 34
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <2 x i64>@llvm.ppc.vsx.xvxexpdp(<2 x double>)

; Function Attrs: nounwind readnone
define <4 x i32> @testXVXSIGSP(<4 x float> %a) {
entry:
  %0 = tail call <4 x i32> @llvm.ppc.vsx.xvxsigsp(<4 x float> %a)
  ret <4 x i32> %0
; CHECK-LABEL: testXVXSIGSP
; CHECK: xvxsigsp 34, 34
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.vsx.xvxsigsp(<4 x float>)

; Function Attrs: nounwind readnone
define <2 x i64> @testXVXSIGDP(<2 x double> %a) {
entry:
  %0 = tail call <2 x i64> @llvm.ppc.vsx.xvxsigdp(<2 x double> %a)
  ret <2 x i64> %0
; CHECK-LABEL: testXVXSIGDP
; CHECK: xvxsigdp 34, 34
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.vsx.xvxsigdp(<2 x double>)

; Function Attrs: nounwind readnone
define <4 x i32> @testXVTSTDCSP(<4 x float> %a) {
entry:
  %0 = tail call <4 x i32> @llvm.ppc.vsx.xvtstdcsp(<4 x float> %a, i32 127)
  ret <4 x i32> %0
; CHECK-LABEL: testXVTSTDCSP
; CHECK: xvtstdcsp 34, 34, 127
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.vsx.xvtstdcsp(<4 x float> %a, i32 %b)

; Function Attrs: nounwind readnone
define <2 x i64> @testXVTSTDCDP(<2 x double> %a) {
entry:
  %0 = tail call <2 x i64> @llvm.ppc.vsx.xvtstdcdp(<2 x double> %a, i32 127)
  ret <2 x i64> %0
; CHECK-LABEL: testXVTSTDCDP
; CHECK: xvtstdcdp 34, 34, 127
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.vsx.xvtstdcdp(<2 x double> %a, i32 %b)

define <4 x float> @testXVCVHPSP(<8 x i16> %a) {
entry:
  %0 = tail call <4 x float>@llvm.ppc.vsx.xvcvhpsp(<8 x i16> %a)
  ret <4 x float> %0
; CHECK-LABEL: testXVCVHPSP
; CHECK: xvcvhpsp 34, 34
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <4 x float>@llvm.ppc.vsx.xvcvhpsp(<8 x i16>)

; Function Attrs: nounwind readnone
define <4 x i32> @testLXVL(i8* %a, i64 %b) {
entry:
  %0 = tail call <4 x i32> @llvm.ppc.vsx.lxvl(i8* %a, i64 %b)
  ret <4 x i32> %0
; CHECK-LABEL: testLXVL
; CHECK: lxvl 34, 3, 4
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.vsx.lxvl(i8*, i64)

define void @testSTXVL(<4 x i32> %a, i8* %b, i64 %c) {
entry:
  tail call void @llvm.ppc.vsx.stxvl(<4 x i32> %a, i8* %b, i64 %c)
  ret void
; CHECK-LABEL: testSTXVL
; CHECK: stxvl 34, 5, 6
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare void @llvm.ppc.vsx.stxvl(<4 x i32>, i8*, i64)

; Function Attrs: nounwind readnone
define <4 x i32> @testLXVLL(i8* %a, i64 %b) {
entry:
  %0 = tail call <4 x i32> @llvm.ppc.vsx.lxvll(i8* %a, i64 %b)
  ret <4 x i32> %0
; CHECK-LABEL: testLXVLL
; CHECK: lxvll 34, 3, 4
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.vsx.lxvll(i8*, i64)

define void @testSTXVLL(<4 x i32> %a, i8* %b, i64 %c) {
entry:
  tail call void @llvm.ppc.vsx.stxvll(<4 x i32> %a, i8* %b, i64 %c)
  ret void
; CHECK-LABEL: testSTXVLL
; CHECK: stxvll 34, 5, 6
; CHECK: blr
}
; Function Attrs: nounwind readnone
declare void @llvm.ppc.vsx.stxvll(<4 x i32>, i8*, i64)

define <4 x i32> @test0(<4 x i32> %a) local_unnamed_addr #0 {
entry:
  %sub.i = sub <4 x i32> zeroinitializer, %a
  ret <4 x i32> %sub.i

; CHECK-LABEL: @test0
; CHECK: vnegw 2, 2
; CHECK: blr

}

define <2 x i64> @test1(<2 x i64> %a) local_unnamed_addr #0 {
entry:
  %sub.i = sub <2 x i64> zeroinitializer, %a
  ret <2 x i64> %sub.i

; CHECK-LABEL: @test1
; CHECK: vnegd 2, 2
; CHECK: blr

}

declare void @sink(...)

; stack object should be accessed using D-form load/store instead of X-form
define signext i32 @func1() {
; CHECK-LABEL: @func1
; CHECK-NOT: stxvx
; CHECK: stxv {{[0-9]+}}, {{[0-9]+}}(1)
; CHECK-NOT: stxvx
; CHECK: blr
entry:
  %a = alloca [4 x i32], align 4
  %0 = bitcast [4 x i32]* %a to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull align 4 %0, i8 0, i64 16, i1 false)
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %a, i64 0, i64 0
  %call = call signext i32 @callee(i32* nonnull %arraydecay) #3
  ret i32 %call
}

; stack object should be accessed using D-form load/store instead of X-form
define signext i32 @func2() {
; CHECK-LABEL: @func2
; CHECK-NOT: stxvx
; CHECK: stxv [[ZEROREG:[0-9]+]], {{[0-9]+}}(1)
; CHECK: stxv [[ZEROREG]], {{[0-9]+}}(1)
; CHECK: stxv [[ZEROREG]], {{[0-9]+}}(1)
; CHECK: stxv [[ZEROREG]], {{[0-9]+}}(1)
; CHECK-NOT: stxvx
; CHECK: blr
entry:
  %a = alloca [16 x i32], align 4
  %0 = bitcast [16 x i32]* %a to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull align 4 %0, i8 0, i64 64, i1 false)
  %arraydecay = getelementptr inbounds [16 x i32], [16 x i32]* %a, i64 0, i64 0
  %call = call signext i32 @callee(i32* nonnull %arraydecay) #3
  ret i32 %call
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1
declare signext i32 @callee(i32*) local_unnamed_addr #2
