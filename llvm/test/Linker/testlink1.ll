; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < %p/testlink2.ll > %t2.bc
; RUN: llvm-link %t.bc %t2.bc

@MyVar = external global i32		; <i32*> [#uses=3]
@MyIntList = global { \2*, i32 } { { \2*, i32 }* null, i32 17 }		; <{ \2*, i32 }*> [#uses=1]
external global i32		; <i32*>:0 [#uses=0]
@Inte = global i32 1		; <i32*> [#uses=0]
@AConst = linkonce constant i32 123		; <i32*> [#uses=0]
@Intern1 = internal constant i32 42		; <i32*> [#uses=0]
@Intern2 = internal constant i32 792		; <i32*> [#uses=0]
@MyVarPtr = linkonce global { i32* } { i32* @MyVar }		; <{ i32* }*> [#uses=0]

declare i32 @foo(i32)

declare void @print(i32)

define void @main() {
	%v1 = load i32* @MyVar		; <i32> [#uses=1]
	call void @print( i32 %v1 )
	%idx = getelementptr { \2*, i32 }* @MyIntList, i64 0, i32 1		; <i32*> [#uses=2]
	%v2 = load i32* %idx		; <i32> [#uses=1]
	call void @print( i32 %v2 )
	call i32 @foo( i32 5 )		; <i32>:1 [#uses=0]
	%v3 = load i32* @MyVar		; <i32> [#uses=1]
	call void @print( i32 %v3 )
	%v4 = load i32* %idx		; <i32> [#uses=1]
	call void @print( i32 %v4 )
	ret void
}

define internal void @testintern() {
	ret void
}

define internal void @Testintern() {
	ret void
}

define void @testIntern() {
	ret void
}
