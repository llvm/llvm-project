; This test ensures that we get a bitcast constant expression in and out,
; not a sitofp constant expression. 
; RUN: llvm-as < %s | llvm-dis | \
; RUN:   grep {bitcast (}

@G = external global i32

define float @tryit(i32 %A) {
   ret float bitcast( i32 ptrtoint (i32* @G to i32) to float)
}
