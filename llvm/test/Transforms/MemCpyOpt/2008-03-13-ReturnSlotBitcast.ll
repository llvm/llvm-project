; RUN: llvm-as < %s | opt -memcpyopt | llvm-dis | not grep {call.*memcpy.}
	%a = type { i32 }
	%b = type { float }

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 
declare void @g(%a*)

define float @f() {
entry:
	%a_var = alloca %a
	%b_var = alloca %b
	call void @g(%a *%a_var)
	%a_i8 = bitcast %a* %a_var to i8*
	%b_i8 = bitcast %b* %b_var to i8*
	call void @llvm.memcpy.i32(i8* %b_i8, i8* %a_i8, i32 4, i32 4)
	%tmp1 = getelementptr %b* %b_var, i32 0, i32 0
	%tmp2 = load float* %tmp1
	ret float %tmp2
}
