; RUN: llc -verify-machineinstrs < %s > %t
; RUN: grep  __floattitf %t
; RUN: grep  __fixunstfti %t

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc64-unknown-linux-gnu.2.0"

define ppc_fp128 @foo(i128 %a) nounwind  {
entry:
	%tmp2829 = uitofp i128 %a to ppc_fp128		; <i64> [#uses=1]
	ret ppc_fp128 %tmp2829
}
define i128 @boo(ppc_fp128 %a) nounwind  {
entry:
	%tmp2829 = fptoui ppc_fp128 %a to i128		; <i64> [#uses=1]
	ret i128 %tmp2829
}
