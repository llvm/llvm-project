; Based llvm/test/CodeGen/X86/aliases.ll
; RUN: llc < %s -mtriple=wasm32-unknown-uknown -asm-verbose=false | FileCheck %s

@bar = global i32 42

; CHECK-DAG: .globl	foo1
; CHECK-DAG: .set foo1, bar
@foo1 = alias i32, ptr @bar

; CHECK-DAG: .globl	foo2
; CHECK-DAG: .set foo2, bar
@foo2 = alias i32, ptr @bar

%FunTy = type i32()

define i32 @foo_f() {
  ret i32 0
}

; CHECK-DAG: .weak	bar_f
; CHECK-DAG: .type	bar_f,@function
; CHECK-DAG: .set bar_f, foo_f
@bar_f = weak alias %FunTy, ptr @foo_f

; CHECK-DAG: .weak	bar_l
; CHECK-DAG: .set bar_l, bar
@bar_l = linkonce_odr alias i32, ptr @bar

; CHECK-DAG: .set bar_i, bar
@bar_i = internal alias i32, ptr @bar

; CHECK-DAG: .globl	A
@A = alias i64, ptr @bar

; CHECK-DAG: .globl	bar_h
; CHECK-DAG: .hidden	bar_h
; CHECK-DAG: .set bar_h, bar
@bar_h = hidden alias i32, ptr @bar

; CHECK-DAG: .globl	bar_p
; CHECK-DAG: .protected	bar_p
; CHECK-DAG: .set bar_p, bar
@bar_p = protected alias i32, ptr @bar

; CHECK-DAG: .set test2, bar+4
@test2 = alias i32, getelementptr(i32, ptr @bar, i32 1)

; CHECK-DAG: .set test3, 42
@test3 = alias i32, inttoptr(i32 42 to ptr)

; CHECK-DAG: .set test4, bar
@test4 = alias i32, inttoptr(i64 ptrtoint (ptr @bar to i64) to ptr)

; CHECK-DAG: .set test5, test2-bar
@test5 = alias i32, inttoptr(i32 sub (i32 ptrtoint (ptr @test2 to i32),
                                 i32 ptrtoint (ptr @bar to i32)) to ptr)

; CHECK-DAG: .globl	test
define i32 @test() {
entry:
   %tmp = load i32, ptr @foo1
   %tmp1 = load i32, ptr @foo2
   %tmp0 = load i32, ptr @bar_i
   %tmp2 = call i32 @foo_f()
   %tmp3 = add i32 %tmp, %tmp2
   %tmp4 = call i32 @bar_f()
   %tmp5 = add i32 %tmp3, %tmp4
   %tmp6 = add i32 %tmp1, %tmp5
   %tmp7 = add i32 %tmp6, %tmp0
   ret i32 %tmp7
}
