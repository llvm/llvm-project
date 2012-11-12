; RUN: llc < %s -march=ppc32

define void @t() nounwind {
	call void null( ppc_fp128 undef )
	unreachable
}
