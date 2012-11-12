; RUN: llc < %s -mtriple=x86_64-apple-darwin -relocation-model=static | not grep lea
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu -relocation-model=static | not grep lea

@v = external global i32, align 8
@v_addr = external global i32*, align 8

define void @t() nounwind optsize {
	store i32* @v, i32** @v_addr, align 8
	unreachable
}
