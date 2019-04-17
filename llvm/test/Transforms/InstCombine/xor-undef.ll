; RUN: opt < %s -instcombine -S | grep zeroinitializer

define <2 x i64> @f() {
	%tmp = xor <2 x i64> undef, undef
        ret <2 x i64> %tmp
}
