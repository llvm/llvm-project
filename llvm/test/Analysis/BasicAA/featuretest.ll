; This testcase tests for various features the basicaa test should be able to 
; determine, as noted in the comments.

; RUN: llvm-as < %s | opt -basicaa -gvn -instcombine -dce | llvm-dis | not grep REMOVE

@Global = external global { i32 }

; Array test:  Test that operations on one local array do not invalidate 
; operations on another array.  Important for scientific codes.
;
define i32 @different_array_test(i64 %A, i64 %B) {
	%Array1 = alloca i32, i32 100
	%Array2 = alloca i32, i32 200

	%pointer = getelementptr i32* %Array1, i64 %A
	%val = load i32* %pointer

	%pointer2 = getelementptr i32* %Array2, i64 %B
	store i32 7, i32* %pointer2

	%REMOVE = load i32* %pointer ; redundant with above load
	%retval = sub i32 %REMOVE, %val
	ret i32 %retval
}

; Constant index test: Constant indexes into the same array should not 
; interfere with each other.  Again, important for scientific codes.
;
define i32 @constant_array_index_test() {
	%Array = alloca i32, i32 100
	%P1 = getelementptr i32* %Array, i64 7
	%P2 = getelementptr i32* %Array, i64 6
	
	%A = load i32* %P1
	store i32 1, i32* %P2   ; Should not invalidate load
	%BREMOVE = load i32* %P1
	%Val = sub i32 %A, %BREMOVE
	ret i32 %Val
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
}

; Test that we can do funny pointer things and that distance calc will still 
; work.
define i32 @gep_distance_test3(i32 * %A) {
	%X = load i32* %A
	%B = bitcast i32* %A to i8*
	%C = getelementptr i8* %B, i64 4
	%Y = load i8* %C
	ret i32 8
}

; Test that we can disambiguate globals reached through constantexpr geps
define i32 @constexpr_test() {
   %X = alloca i32
   %Y = load i32* %X
   store i32 5, i32* getelementptr ({ i32 }* @Global, i64 0, i32 0)
   %REMOVE = load i32* %X
   %retval = sub i32 %Y, %REMOVE
   ret i32 %retval
}
