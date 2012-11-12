; RUN: llc < %s -march=x86 | grep rdtsc
; RUN: llc < %s -march=x86-64 | grep rdtsc
declare i64 @llvm.readcyclecounter()

define i64 @foo() {
	%tmp.1 = call i64 @llvm.readcyclecounter( )		; <i64> [#uses=1]
	ret i64 %tmp.1
}
