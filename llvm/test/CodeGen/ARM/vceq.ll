; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vceqi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vceqi8:
;CHECK: vceq.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = icmp eq <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vceqi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vceqi16:
;CHECK: vceq.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp eq <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vceqi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vceqi32:
;CHECK: vceq.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = icmp eq <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <2 x i32> @vceqf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vceqf32:
;CHECK: vceq.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = fcmp oeq <2 x float> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <16 x i8> @vceqQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vceqQi8:
;CHECK: vceq.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = icmp eq <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vceqQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vceqQi16:
;CHECK: vceq.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = icmp eq <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vceqQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vceqQi32:
;CHECK: vceq.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = icmp eq <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <4 x i32> @vceqQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vceqQf32:
;CHECK: vceq.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = fcmp oeq <4 x float> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <8 x i8> @vceqi8Z(<8 x i8>* %A) nounwind {
;CHECK: vceqi8Z:
;CHECK-NOT: vmov
;CHECK-NOT: vmvn
;CHECK: vceq.i8
	%tmp1 = load <8 x i8>* %A
	%tmp3 = icmp eq <8 x i8> %tmp1, <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}
