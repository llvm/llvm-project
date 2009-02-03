; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=arm-apple-darwin -relocation-model=static | \
; RUN:   not grep {L_G\$non_lazy_ptr}
; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=arm-apple-darwin -relocation-model=dynamic-no-pic | \
; RUN:   grep {L_G\$non_lazy_ptr} | count 2
; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=arm-apple-darwin -relocation-model=pic | \
; RUN:   grep {ldr.*pc} | count 1
; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=arm-linux-gnueabi -relocation-model=pic | \
; RUN:   grep {GOT} | count 1

@G = external global i32

define i32 @test1() {
	%tmp = load i32* @G
	ret i32 %tmp
}
