; RUN: opt -passes=adce -S < %s | not grep call

declare i32 @strlen(i8*) readonly nounwind willreturn

define void @test() {
	call i32 @strlen( i8* null )		; <i32>:1 [#uses=0]
	ret void
}
