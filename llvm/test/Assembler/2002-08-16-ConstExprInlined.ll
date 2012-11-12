; In this testcase, the bytecode reader or writer is not correctly handling the
; ConstExpr reference.  Disassembling this program assembled yields invalid
; assembly (because there are placeholders still around), which the assembler
; dies on.

; There are two things that need to be fixed here.  Obviously assembling and
; disassembling this would be good, but in addition to that, the bytecode
; reader should NEVER produce a program "successfully" with placeholders still
; around!
;
; RUN: llvm-as < %s | llvm-dis | llvm-as

@.LC0 = internal global [4 x i8] c"foo\00"		; <[4 x i8]*> [#uses=1]
@X = global i8* null		; <i8**> [#uses=0]

declare i32 @puts(i8*)

define void @main() {
bb1:
	%reg211 = call i32 @puts( i8* getelementptr ([4 x i8]* @.LC0, i64 0, i64 0) )		; <i32> [#uses=0]
	ret void
}
