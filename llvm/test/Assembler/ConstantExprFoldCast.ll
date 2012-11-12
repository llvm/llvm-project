; This test checks to make sure that constant exprs fold in some simple situations

; RUN: llvm-as < %s | llvm-dis | not grep cast

@A = global i32* bitcast (i8* null to i32*)  ; Cast null -> fold
@B = global i32** bitcast (i32** @A to i32**)   ; Cast to same type -> fold
@C = global i32 trunc (i64 42 to i32)        ; Integral casts
@D = global i32* bitcast(float*  bitcast (i32* @C to float*) to i32*)  ; cast of cast ptr->ptr
@E = global i32 ptrtoint(float* inttoptr (i8 5 to float*) to i32)  ; i32 -> ptr -> i32

; Test folding of binary instrs
@F = global i32* inttoptr (i32 add (i32 5, i32 -5) to i32*)
@G = global i32* inttoptr (i32 sub (i32 5, i32 5) to i32*)

