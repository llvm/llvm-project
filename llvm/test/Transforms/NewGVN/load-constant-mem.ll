; RUN: opt < %s -basicaa -newgvn -S | FileCheck %s
; PR4189
@G = external constant [4 x i32]

define i32 @test(i8* %p, i32 %i) nounwind {
entry:
	%P = getelementptr [4 x i32], [4 x i32]* @G, i32 0, i32 %i
	%A = load i32, i32* %P
	store i8 4, i8* %p
	%B = load i32, i32* %P
	%C = sub i32 %A, %B
	ret i32 %C
}

; CHECK-LABEL: define i32 @test
; CHECK-NEXT: entry:
; CHECK-NEXT:   %P = getelementptr [4 x i32], [4 x i32]* @G, i32 0, i32 %i
; CHECK-NEXT:   %A = load i32, i32* %P
; CHECK-NEXT:   store i8 4, i8* %p
; CHECK-NEXT:   ret i32 0
; CHECK-NEXT: }
