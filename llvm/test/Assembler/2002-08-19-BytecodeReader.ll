; Testcase that seems to break the bytecode reader.  This comes from the
; "crafty" spec benchmark.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | llvm-as
	
%CHESS_POSITION = type { i32, i32 }
@pawn_probes = external global i32		; <i32*> [#uses=0]
@pawn_hash_mask = external global i32		; <i32*> [#uses=0]
@search = external global %CHESS_POSITION		; <%CHESS_POSITION*> [#uses=2]

define void @Evaluate() {
	%reg1321 = getelementptr %CHESS_POSITION* @search, i64 0, i32 1		; <i32*> [#uses=1]
	%reg114 = load i32* %reg1321		; <i32> [#uses=0]
	%reg1801 = getelementptr %CHESS_POSITION* @search, i64 0, i32 0		; <i32*> [#uses=1]
	%reg182 = load i32* %reg1801		; <i32> [#uses=0]
	ret void
}
