; RUN: llvm-as < %s | llc

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8.8.0"


define void @blargh() {
entry:
	%tmp4 = call i32 asm "rlwimi $0,$2,$3,$4,$5", "=r,0,r,n,n,n"( i32 0, i32 0, i32 0, i32 24, i32 31 )		; <i32> [#uses=0]
	unreachable
}
