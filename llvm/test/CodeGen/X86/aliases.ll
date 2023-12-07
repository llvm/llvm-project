; RUN: llc < %s -mtriple=i686-pc-linux-gnu -asm-verbose=false \
; RUN: -relocation-model=pic | FileCheck %s

@thread_var = thread_local global i32 42, align 4
@thread_alias = thread_local(localdynamic) alias i32, ptr @thread_var

; CHECK-LABEL: get_thread_var
define ptr @get_thread_var() {
; CHECK: leal    thread_var@TLSGD
  ret ptr @thread_var
}

; CHECK-LABEL: get_thread_alias
define ptr @get_thread_alias() {
; CHECK: leal    thread_alias@TLSLD
  ret ptr @thread_alias
}

@bar = global i32 42

; CHECK-DAG: .globl	foo1
@foo1 = alias i32, ptr @bar

; CHECK-DAG: .globl	foo2
@foo2 = alias i32, ptr @bar

%FunTy = type i32()

define i32 @foo_f() {
  ret i32 0
}
; CHECK-DAG: .weak	bar_f
@bar_f = weak alias %FunTy, ptr @foo_f

@bar_l = linkonce_odr alias i32, ptr @bar
; CHECK-DAG: .weak	bar_l

@bar_i = internal alias i32, ptr @bar

; CHECK-DAG: .globl	A
@A = alias i64, ptr @bar

; CHECK-DAG: .globl	bar_h
; CHECK-DAG: .hidden	bar_h
@bar_h = hidden alias i32, ptr @bar

; CHECK-DAG: .globl	bar_p
; CHECK-DAG: .protected	bar_p
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
