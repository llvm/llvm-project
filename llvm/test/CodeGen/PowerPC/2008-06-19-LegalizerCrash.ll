; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--

define void @t() nounwind {
	call void null( ppc_fp128 undef )
	unreachable
}
