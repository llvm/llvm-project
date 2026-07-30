; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--

@g = external global ppc_fp128
@h = external global ppc_fp128

define void @f() {
	%tmp = load ppc_fp128, ptr @g
	store ppc_fp128 %tmp, ptr @h
	ret void
}
