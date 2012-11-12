; This test shows an alloca of a struct and an array that can be reduced to
; multiple variables easily. However, the alloca is used by a store
; instruction, which was not possible before aggregrates were first class
; values. This checks of scalarrepl splits up the struct and array properly.

; RUN: opt < %s -scalarrepl -S | not grep alloca
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

define i32 @foo() {
	%target = alloca { i32, i32 }		; <{ i32, i32 }*> [#uses=1]
        ; Build a first class struct to store
	%res1 = insertvalue { i32, i32 } undef, i32 1, 0		; <{ i32, i32 }> [#uses=1]
	%res2 = insertvalue { i32, i32 } %res1, i32 2, 1		; <{ i32, i32 }> [#uses=1]
        ; And store it
	store { i32, i32 } %res2, { i32, i32 }* %target
        ; Actually use %target, so it doesn't get removed altogether
        %ptr = getelementptr { i32, i32 }* %target, i32 0, i32 0
        %val = load i32* %ptr
	ret i32 %val
}

define i32 @bar() {
	%target = alloca [ 2 x i32 ]		; <{ i32, i32 }*> [#uses=1]
        ; Build a first class array to store
	%res1 = insertvalue [ 2 x i32 ] undef, i32 1, 0		; <{ i32, i32 }> [#uses=1]
	%res2 = insertvalue [ 2 x i32 ] %res1, i32 2, 1		; <{ i32, i32 }> [#uses=1]
        ; And store it
	store [ 2 x i32 ] %res2, [ 2 x i32 ]* %target
        ; Actually use %target, so it doesn't get removed altogether
        %ptr = getelementptr [ 2 x i32 ]* %target, i32 0, i32 0
        %val = load i32* %ptr
	ret i32 %val
}
