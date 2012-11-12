; This testcase tests for various features the basicaa test should be able to 
; determine, as noted in the comments.

; RUN: opt < %s -basicaa -gvn -instcombine -dce -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@Global = external global { i32 }

declare void @external(i32*)

; Array test:  Test that operations on one local array do not invalidate 
; operations on another array.  Important for scientific codes.
;
define i32 @different_array_test(i64 %A, i64 %B) {
	%Array1 = alloca i32, i32 100
	%Array2 = alloca i32, i32 200
        
        call void @external(i32* %Array1)
        call void @external(i32* %Array2)

	%pointer = getelementptr i32* %Array1, i64 %A
	%val = load i32* %pointer

	%pointer2 = getelementptr i32* %Array2, i64 %B
	store i32 7, i32* %pointer2

	%REMOVE = load i32* %pointer ; redundant with above load
	%retval = sub i32 %REMOVE, %val
	ret i32 %retval
; CHECK: @different_array_test
; CHECK: ret i32 0
}

; Constant index test: Constant indexes into the same array should not 
; interfere with each other.  Again, important for scientific codes.
;
define i32 @constant_array_index_test() {
	%Array = alloca i32, i32 100
        call void @external(i32* %Array)

	%P1 = getelementptr i32* %Array, i64 7
	%P2 = getelementptr i32* %Array, i64 6
	
	%A = load i32* %P1
	store i32 1, i32* %P2   ; Should not invalidate load
	%BREMOVE = load i32* %P1
	%Val = sub i32 %A, %BREMOVE
	ret i32 %Val
; CHECK: @constant_array_index_test
; CHECK: ret i32 0
}

; Test that if two pointers are spaced out by a constant getelementptr, that 
; they cannot alias.
define i32 @gep_distance_test(i32* %A) {
        %REMOVEu = load i32* %A
        %B = getelementptr i32* %A, i64 2  ; Cannot alias A
        store i32 7, i32* %B
        %REMOVEv = load i32* %A
        %r = sub i32 %REMOVEu, %REMOVEv
        ret i32 %r
; CHECK: @gep_distance_test
; CHECK: ret i32 0
}

; Test that if two pointers are spaced out by a constant offset, that they
; cannot alias, even if there is a variable offset between them...
define i32 @gep_distance_test2({i32,i32}* %A, i64 %distance) {
	%A1 = getelementptr {i32,i32}* %A, i64 0, i32 0
	%REMOVEu = load i32* %A1
	%B = getelementptr {i32,i32}* %A, i64 %distance, i32 1
	store i32 7, i32* %B    ; B cannot alias A, it's at least 4 bytes away
	%REMOVEv = load i32* %A1
        %r = sub i32 %REMOVEu, %REMOVEv
        ret i32 %r
; CHECK: @gep_distance_test2
; CHECK: ret i32 0
}

; Test that we can do funny pointer things and that distance calc will still 
; work.
define i32 @gep_distance_test3(i32 * %A) {
	%X = load i32* %A
	%B = bitcast i32* %A to i8*
	%C = getelementptr i8* %B, i64 4
        store i8 42, i8* %C
	%Y = load i32* %A
        %R = sub i32 %X, %Y
	ret i32 %R
; CHECK: @gep_distance_test3
; CHECK: ret i32 0
}

; Test that we can disambiguate globals reached through constantexpr geps
define i32 @constexpr_test() {
   %X = alloca i32
   call void @external(i32* %X)

   %Y = load i32* %X
   store i32 5, i32* getelementptr ({ i32 }* @Global, i64 0, i32 0)
   %REMOVE = load i32* %X
   %retval = sub i32 %Y, %REMOVE
   ret i32 %retval
; CHECK: @constexpr_test
; CHECK: ret i32 0
}



; PR7589
; These two index expressions are different, this cannot be CSE'd.
define i16 @zext_sext_confusion(i16* %row2col, i5 %j) nounwind{
entry:
  %sum5.cast = zext i5 %j to i64             ; <i64> [#uses=1]
  %P1 = getelementptr i16* %row2col, i64 %sum5.cast
  %row2col.load.1.2 = load i16* %P1, align 1 ; <i16> [#uses=1]
  
  %sum13.cast31 = sext i5 %j to i6          ; <i6> [#uses=1]
  %sum13.cast = zext i6 %sum13.cast31 to i64      ; <i64> [#uses=1]
  %P2 = getelementptr i16* %row2col, i64 %sum13.cast
  %row2col.load.1.6 = load i16* %P2, align 1 ; <i16> [#uses=1]
  
  %.ret = sub i16 %row2col.load.1.6, %row2col.load.1.2 ; <i16> [#uses=1]
  ret i16 %.ret
; CHECK: @zext_sext_confusion
; CHECK: ret i16 %.ret
}
