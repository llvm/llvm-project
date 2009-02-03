; RUN: llvm-as < %s | llc -march=ppc64 | grep __floatditf

define i64 @__fixunstfdi(ppc_fp128 %a) nounwind  {
entry:
	%tmp1213 = uitofp i64 0 to ppc_fp128		; <ppc_fp128> [#uses=1]
	%tmp15 = sub ppc_fp128 %a, %tmp1213		; <ppc_fp128> [#uses=1]
	%tmp2829 = fptoui ppc_fp128 %tmp15 to i32		; <i32> [#uses=1]
	%tmp282930 = zext i32 %tmp2829 to i64		; <i64> [#uses=1]
	%tmp32 = add i64 %tmp282930, 0		; <i64> [#uses=1]
	ret i64 %tmp32
}
