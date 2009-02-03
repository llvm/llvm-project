; This file is used by testlink1.ll, so it doesn't actually do anything itself
;
; RUN: true

@MyVar = global i32 4		; <i32*> [#uses=2]
@MyIntList = external global { \2*, i32 }		; <{ \2*, i32 }*> [#uses=2]
@AConst = constant i32 123		; <i32*> [#uses=0]

;; Intern in both testlink[12].ll
@Intern1 = internal constant i32 52		; <i32*> [#uses=0]

;; Intern in one but not in other
@Intern2 = constant i32 12345		; <i32*> [#uses=0]

@MyIntListPtr = constant { { \2*, i32 }* } { { \2*, i32 }* @MyIntList }		; <{ { \2*, i32 }* }*> [#uses=0]
@MyVarPtr = linkonce global { i32* } { i32* @MyVar }		; <{ i32* }*> [#uses=0]
constant i32 412		; <i32*>:0 [#uses=1]

define i32 @foo(i32 %blah) {
	store i32 %blah, i32* @MyVar
	%idx = getelementptr { \2*, i32 }* @MyIntList, i64 0, i32 1		; <i32*> [#uses=1]
	store i32 12, i32* %idx
	%ack = load i32* @0		; <i32> [#uses=1]
	%fzo = add i32 %ack, %blah		; <i32> [#uses=1]
	ret i32 %fzo
}

declare void @unimp(float, double)

define internal void @testintern() {
	ret void
}

define void @Testintern() {
	ret void
}

define internal void @testIntern() {
	ret void
}

