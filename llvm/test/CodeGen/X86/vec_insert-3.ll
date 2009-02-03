; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+sse2,-sse41 | grep punpcklqdq | count 1

define <2 x i64> @t1(i64 %s, <2 x i64> %tmp) nounwind {
        %tmp1 = insertelement <2 x i64> %tmp, i64 %s, i32 1
        ret <2 x i64> %tmp1
}
