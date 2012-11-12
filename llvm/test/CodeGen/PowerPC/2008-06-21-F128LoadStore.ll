; RUN: llc < %s -march=ppc32

@g = external global ppc_fp128
@h = external global ppc_fp128

define void @f() {
	%tmp = load ppc_fp128* @g
	store ppc_fp128 %tmp, ppc_fp128* @h
	ret void
}
