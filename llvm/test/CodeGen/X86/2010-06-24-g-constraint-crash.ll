; RUN: llc %s -mtriple=x86_64-apple-darwin10 -frame-pointer=all -o /dev/null
; Formerly crashed, rdar://8015842

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%0 = type { i64, i64, i64, i64, i64 }

@utcbs.1559 = internal global [3 x i64] zeroinitializer ; <ptr> [#uses=1]

define void @bar() nounwind ssp {
entry:
  %asmtmp.i.i = tail call %0 asm sideeffect "push %rbp; syscall; pop %rbp\0A", "={ax},={di},={si},={dx},={bx},{ax},{di},{si},{dx},{bx},~{dirflag},~{fpsr},~{flags},~{memory},~{r15},~{r14},~{r13},~{r12},~{r11},~{r10},~{r9},~{r8},~{rcx}"(i32 7, i64 -1, i64 0, i64 -1, i64 -1) nounwind ; <%0> [#uses=0]
  %asmtmp.i1.i = tail call %0 asm sideeffect "mov $10, %r8;\0Amov $11, %r9;\0Amov $12, %r10;\0Apush %rbp; syscall; pop %rbp\0A", "={ax},={di},={si},={dx},={bx},{ax},{di},{si},{dx},{bx},imr,imr,imr,~{dirflag},~{fpsr},~{flags},~{memory},~{r15},~{r14},~{r13},~{r12},~{r11},~{r10},~{r9},~{r8},~{rcx}"(i32 8, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 0, ptr getelementptr inbounds ([3 x i64], ptr @utcbs.1559, i64 0, i64 2)) nounwind ; <%0> [#uses=0]
  ret void
}
