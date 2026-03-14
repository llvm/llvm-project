; Test the vector rotate and shift doubleword instructions that were added in P8
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s

declare <2 x i64> @llvm.ppc.altivec.vrld(<2 x i64>, <2 x i64>) nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.vsld(<2 x i64>, <2 x i64>) nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.vsrd(<2 x i64>, <2 x i64>) nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.vsrad(<2 x i64>, <2 x i64>) nounwind readnone

define <2 x i64> @test_vrld(<2 x i64> %x, <2 x i64> %y) nounwind readnone {
       %tmp = tail call <2 x i64> @llvm.ppc.altivec.vrld(<2 x i64> %x, <2 x i64> %y)
       ret <2 x i64> %tmp
; CHECK: vrld 2, 2, 3
}

define <2 x i64> @test_vsld(<2 x i64> %x, <2 x i64> %y) nounwind readnone {
       %tmp = shl <2 x i64> %x, %y
       ret <2 x i64> %tmp
; CHECK-LABEL: @test_vsld
; CHECK: vsld 2, 2, 3
}

define <2 x i64> @test_vsrd(<2 x i64> %x, <2 x i64> %y) nounwind readnone {
	%tmp = lshr <2 x i64> %x, %y
	ret <2 x i64> %tmp
; CHECK-LABEL: @test_vsrd
; CHECK: vsrd 2, 2, 3
}

define <2 x i64> @test_vsrad(<2 x i64> %x, <2 x i64> %y) nounwind readnone {
	%tmp = ashr <2 x i64> %x, %y
	ret <2 x i64> %tmp
; CHECK-LABEL: @test_vsrad
; CHECK: vsrad 2, 2, 3
}
       
