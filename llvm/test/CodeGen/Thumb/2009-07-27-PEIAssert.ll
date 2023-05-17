; RUN: llc < %s -mtriple=thumbv6-apple-darwin -relocation-model=pic -frame-pointer=all

	%struct.LinkList = type { i32, ptr }
	%struct.List = type { i32, ptr }
@llvm.used = appending global [1 x ptr] [ptr @main], section "llvm.metadata"		; <ptr> [#uses=0]

define i32 @main() nounwind {
entry:
	%ll = alloca ptr, align 4		; <ptr> [#uses=1]
	%0 = call  i32 @ReadList(ptr %ll, ptr null) nounwind		; <i32> [#uses=1]
	switch i32 %0, label %bb5 [
		i32 7, label %bb4
		i32 42, label %bb3
	]

bb3:		; preds = %entry
	ret i32 1

bb4:		; preds = %entry
	ret i32 0

bb5:		; preds = %entry
	ret i32 1
}

declare i32 @ReadList(ptr nocapture, ptr nocapture) nounwind
