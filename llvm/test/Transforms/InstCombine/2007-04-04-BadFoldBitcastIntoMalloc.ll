; In the presence of a negative offset (the -8 below), a fold of a bitcast into
; a malloc messes up the element count, causing an extra 4GB to be allocated on
; 64-bit targets.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {= add }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "x86_64-unknown-freebsd6.2"

define i1 @test(i32 %tmp141, double** %tmp145)
{
  %tmp133 = add i32 %tmp141, 1
  %tmp134 = shl i32 %tmp133, 3
  %tmp135 = add i32 %tmp134, -8
  %tmp136 = malloc i8, i32 %tmp135
  %tmp137 = bitcast i8* %tmp136 to double*
  store double* %tmp137, double** %tmp145
  ret i1 false
}
