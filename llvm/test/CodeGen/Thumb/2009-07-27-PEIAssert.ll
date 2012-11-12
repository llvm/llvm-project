; RUN: llc < %s -mtriple=thumbv6-apple-darwin -relocation-model=pic -disable-fp-elim

	%struct.LinkList = type { i32, %struct.LinkList* }
	%struct.List = type { i32, i32* }
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 ()* @main to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i32 @main() nounwind {
entry:
	%ll = alloca %struct.LinkList*, align 4		; <%struct.LinkList**> [#uses=1]
	%0 = call  i32 @ReadList(%struct.LinkList** %ll, %struct.List** null) nounwind		; <i32> [#uses=1]
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

declare i32 @ReadList(%struct.LinkList** nocapture, %struct.List** nocapture) nounwind
