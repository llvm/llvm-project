; The CBE should not emit code that casts the function pointer.  This causes
; GCC to get testy and insert trap instructions instead of doing the right
; thing. :(
; RUN: llvm-as < %s | llc -march=c

declare void @external(i8*)

define i32 @test(i32* %X) {
        %RV = call i32 bitcast (void (i8*)* @external to i32 (i32*)*)( i32* %X )                ; <i32> [#uses=1]
        ret i32 %RV
}

