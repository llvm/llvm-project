; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep zeroinitializer

define <2 x i64> @f() {
	%tmp = xor <2 x i64> undef, undef
        ret <2 x i64> %tmp
}
