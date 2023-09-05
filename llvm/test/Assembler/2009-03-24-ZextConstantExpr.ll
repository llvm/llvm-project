; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s
; PR3876
@gdtr = external global [0 x i8]

define void @test() {
	call zeroext i1 @paging_map(i64 zext (i32 add (i32 ptrtoint (ptr @gdtr to i32), i32 -4096) to i64))
	ret void
}

declare zeroext i1 @paging_map(i64)

