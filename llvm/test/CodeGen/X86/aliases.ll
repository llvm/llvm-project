; RUN: llc < %s -mtriple=i686-pc-linux-gnu -asm-verbose=false -o %t
; RUN: grep globl %t | count 6
; RUN: grep weak %t  | count 1
; RUN: grep hidden %t | count 1
; RUN: grep protected %t | count 1

@bar = external global i32
@foo1 = alias i32* @bar
@foo2 = alias i32* @bar

%FunTy = type i32()

declare i32 @foo_f()
@bar_f = alias weak %FunTy* @foo_f

@bar_i = alias internal i32* @bar

@A = alias bitcast (i32* @bar to i64*)

@bar_h = hidden alias i32* @bar

@bar_p = protected alias i32* @bar

define i32 @test() {
entry:
   %tmp = load i32* @foo1
   %tmp1 = load i32* @foo2
   %tmp0 = load i32* @bar_i
   %tmp2 = call i32 @foo_f()
   %tmp3 = add i32 %tmp, %tmp2
   %tmp4 = call %FunTy* @bar_f()
   %tmp5 = add i32 %tmp3, %tmp4
   %tmp6 = add i32 %tmp1, %tmp5
   %tmp7 = add i32 %tmp6, %tmp0
   ret i32 %tmp7
}
