; RUN: llc < %s -enable-legalize-types-checking
; PR8582
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i686-pc-win32"

define void @test() nounwind {
 %i17 = icmp eq <4 x i8> undef, zeroinitializer
 %cond = extractelement <4 x i1> %i17, i32 0
 %_comp = select i1 %cond, i8 0, i8 undef
 %merge = insertelement <4 x i8> undef, i8 %_comp, i32 0
 %cond3 = extractelement <4 x i1> %i17, i32 1
 %_comp4 = select i1 %cond3, i8 0, i8 undef
 %merge5 = insertelement <4 x i8> %merge, i8 %_comp4, i32 1
 %cond8 = extractelement <4 x i1> %i17, i32 2
 %_comp9 = select i1 %cond8, i8 0, i8 undef
 %m387 = insertelement <4 x i8> %merge5, i8 %_comp9, i32 2
 store <4 x i8> %m387, <4 x i8>* undef
 ret void
}
