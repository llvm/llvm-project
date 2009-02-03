; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2,-sse41 | grep {\$36,} | count 2
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2,-sse41 | grep shufps | count 2
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2,-sse41 | grep pinsrw | count 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2,-sse41 | grep movhpd | count 1
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+sse2,-sse41 | grep unpcklpd | count 1

define <4 x float> @t1(float %s, <4 x float> %tmp) nounwind {
        %tmp1 = insertelement <4 x float> %tmp, float %s, i32 3
        ret <4 x float> %tmp1
}

define <4 x i32> @t2(i32 %s, <4 x i32> %tmp) nounwind {
        %tmp1 = insertelement <4 x i32> %tmp, i32 %s, i32 3
        ret <4 x i32> %tmp1
}

define <2 x double> @t3(double %s, <2 x double> %tmp) nounwind {
        %tmp1 = insertelement <2 x double> %tmp, double %s, i32 1
        ret <2 x double> %tmp1
}

define <8 x i16> @t4(i16 %s, <8 x i16> %tmp) nounwind {
        %tmp1 = insertelement <8 x i16> %tmp, i16 %s, i32 5
        ret <8 x i16> %tmp1
}
