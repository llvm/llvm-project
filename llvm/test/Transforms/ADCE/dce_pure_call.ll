; RUN: opt -passes=adce -S < %s | not grep call

declare i32 @strlen(ptr) readonly nounwind willreturn

define void @test() {
	call i32 @strlen( ptr null )		; <i32>:1 [#uses=0]
	ret void
}
