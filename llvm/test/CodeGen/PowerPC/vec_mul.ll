; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -march=ppc32 -mattr=+altivec -mattr=-vsx -mattr=-power8-altivec | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -march=ppc64 -mattr=+altivec -mattr=-vsx -mcpu=pwr7 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -march=ppc64 -mattr=+altivec -mattr=-vsx -mcpu=pwr8 -mattr=-power8-altivec | FileCheck %s -check-prefix=CHECK-LE
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -march=ppc64 -mattr=+altivec -mattr=+vsx -mcpu=pwr7 | FileCheck %s -check-prefix=CHECK-VSX
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -march=ppc64 -mattr=+altivec -mattr=+vsx -mcpu=pwr8 -mattr=-power8-altivec | FileCheck %s -check-prefix=CHECK-LE-VSX

define <4 x i32> @test_v4i32(<4 x i32>* %X, <4 x i32>* %Y) {
	%tmp = load <4 x i32>, <4 x i32>* %X		; <<4 x i32>> [#uses=1]
	%tmp2 = load <4 x i32>, <4 x i32>* %Y		; <<4 x i32>> [#uses=1]
	%tmp3 = mul <4 x i32> %tmp, %tmp2		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp3
}
; CHECK-LABEL: test_v4i32:
; CHECK: vmsumuhm
; CHECK-NOT: mullw
; CHECK-LE-LABEL: test_v4i32:
; CHECK-LE: vmsumuhm
; CHECK-LE-NOT: mullw
; CHECK-VSX-LABEL: test_v4i32:
; CHECK-VSX: vmsumuhm
; CHECK-VSX-NOT: mullw
; CHECK-LE-VSX-LABEL: test_v4i32:
; CHECK-LE-VSX: vmsumuhm
; CHECK-LE-VSX-NOT: mullw

define <8 x i16> @test_v8i16(<8 x i16>* %X, <8 x i16>* %Y) {
	%tmp = load <8 x i16>, <8 x i16>* %X		; <<8 x i16>> [#uses=1]
	%tmp2 = load <8 x i16>, <8 x i16>* %Y		; <<8 x i16>> [#uses=1]
	%tmp3 = mul <8 x i16> %tmp, %tmp2		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %tmp3
}
; CHECK-LABEL: test_v8i16:
; CHECK: vmladduhm
; CHECK-NOT: mullw
; CHECK-LE-LABEL: test_v8i16:
; CHECK-LE: vmladduhm
; CHECK-LE-NOT: mullw
; CHECK-VSX-LABEL: test_v8i16:
; CHECK-VSX: vmladduhm
; CHECK-VSX-NOT: mullw
; CHECK-LE-VSX-LABEL: test_v8i16:
; CHECK-LE-VSX: vmladduhm
; CHECK-LE-VSX-NOT: mullw

define <16 x i8> @test_v16i8(<16 x i8>* %X, <16 x i8>* %Y) {
	%tmp = load <16 x i8>, <16 x i8>* %X		; <<16 x i8>> [#uses=1]
	%tmp2 = load <16 x i8>, <16 x i8>* %Y		; <<16 x i8>> [#uses=1]
	%tmp3 = mul <16 x i8> %tmp, %tmp2		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %tmp3
}
; CHECK-LABEL: test_v16i8:
; CHECK: vmuloub
; CHECK: vmuleub
; CHECK-NOT: mullw
; CHECK-LE-LABEL: test_v16i8:
; CHECK-LE: vmuloub [[REG1:[0-9]+]]
; CHECK-LE: vmuleub [[REG2:[0-9]+]]
; CHECK-LE: vperm {{[0-9]+}}, [[REG2]], [[REG1]]
; CHECK-LE-NOT: mullw
; CHECK-VSX-LABEL: test_v16i8:
; CHECK-VSX: vmuloub
; CHECK-VSX: vmuleub
; CHECK-VSX-NOT: mullw
; CHECK-LE-VSX-LABEL: test_v16i8:
; CHECK-LE-VSX: vmuloub [[REG1:[0-9]+]]
; CHECK-LE-VSX: vmuleub [[REG2:[0-9]+]]
; CHECK-LE-VSX: vperm {{[0-9]+}}, [[REG2]], [[REG1]]
; CHECK-LE-VSX-NOT: mullw

define <4 x float> @test_float(<4 x float>* %X, <4 x float>* %Y) {
	%tmp = load <4 x float>, <4 x float>* %X
	%tmp2 = load <4 x float>, <4 x float>* %Y
	%tmp3 = fmul <4 x float> %tmp, %tmp2
	ret <4 x float> %tmp3
}
; Check the creation of a negative zero float vector by creating a vector of
; all bits set and shifting it 31 bits to left, resulting a an vector of 
; 4 x 0x80000000 (-0.0 as float).
; CHECK-LABEL: test_float:
; CHECK: vspltisw [[ZNEG:[0-9]+]], -1
; CHECK: vslw     {{[0-9]+}}, [[ZNEG]], [[ZNEG]]
; CHECK: vmaddfp
; CHECK-LE-LABEL: test_float:
; CHECK-LE: vspltisw [[ZNEG:[0-9]+]], -1
; CHECK-LE: vslw     {{[0-9]+}}, [[ZNEG]], [[ZNEG]]
; CHECK-LE: vmaddfp
; CHECK-VSX-LABEL: test_float:
; CHECK-VSX: xvmulsp
; CHECK-LE-VSX-LABEL: test_float:
; CHECK-LE-VSX: xvmulsp
