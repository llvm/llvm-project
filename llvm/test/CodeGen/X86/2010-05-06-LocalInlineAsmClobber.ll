; RUN: llc -regalloc=fast -optimize-regalloc=0 %s -o %t
; PR7066

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @sys_clone(i32 (i8*)* %fn, i8* %child_stack, i32 %flags, i8* %arg, i32* %parent_tidptr, i8* %newtls, i32* %child_tidptr) nounwind {
  call i64 asm sideeffect "", "={ax},0,i,i,r,{si},{di},r,{dx},imr,imr,~{sp},~{memory},~{r8},~{r10},~{r11},~{cx},~{dirflag},~{fpsr},~{flags}"(i64 4294967274, i32 56, i32 60, i32 (i8*)* undef, i8* undef, i32 undef, i8* undef, i32* undef, i8* undef, i32* undef) nounwind ; <i64> [#uses=0]
  ret i32 undef
}
