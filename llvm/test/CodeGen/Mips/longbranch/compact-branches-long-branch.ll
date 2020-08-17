; RUN: llc < %s -march=mips -mcpu=mips32r6 -force-mips-long-branch | FileCheck %s

; Check that when MIPS32R6 with the static relocation model with the usage of
; long branches, that there is a nop between any compact branch and the static
; relocation method of expanding branches. Previously, it could result in 'j'
; following a b(ne|eq)zc, which would raise a reserved instruction exception.

declare i32 @f(i32)

declare i32 @g()

; CHECK-LABEL: test1:
; CHECK:       bnezc
; CHECK-NEXT:  nop

define i32 @test1(i32 %a) {
entry:
  %0 = icmp eq i32 %a, 0
  br i1 %0, label %cond.true, label %cond.false
cond.true:
  %1 = call i32 @f(i32 %a)
  ret i32 %1
cond.false:
  %2 = call i32 @g()
  ret i32 %2
}

; CHECK-LABEL: test2:
; CHECK:       bgezc
; CHECK-NEXT:  nop

define i32 @test2(i32 %a) {
entry:
  %0 = icmp sge i32 %a, 0
  br i1 %0, label %cond.true, label %cond.false
cond.true:
  %1 = call i32 @f(i32 %a)
  ret i32 %1
cond.false:
  %2 = call i32 @g()
  ret i32 %2
}

; CHECK-LABEL: test3:
; CHECK:       blezc
; CHECK-NEXT:  nop

define i32 @test3(i32 %a) {
entry:
  %0 = icmp sle i32 %a, 0
  br i1 %0, label %cond.true, label %cond.false
cond.true:
  %1 = call i32 @f(i32 %a)
  ret i32 %1
cond.false:
  %2 = call i32 @g()
  ret i32 %2
}

; CHECK-LABEL: test4:
; CHECK:       bgtzc
; CHECK-NEXT:  nop

define i32 @test4(i32 %a) {
entry:
  %0 = icmp sgt i32 %a, 0
  br i1 %0, label %cond.true, label %cond.false
cond.true:
  %1 = call i32 @f(i32 %a)
  ret i32 %1
cond.false:
  %2 = call i32 @g()
  ret i32 %2
}

; CHECK-LABEL: test5:
; CHECK:       bgezc
; CHECK-NEXT:  nop

define i32 @test5(i32 %a) {
entry:
  %0 = icmp slt i32 %a, 0
  br i1 %0, label %cond.true, label %cond.false
cond.true:
  %1 = call i32 @f(i32 %a)
  ret i32 %1
cond.false:
  %2 = call i32 @g()
  ret i32 %2
}

; CHECK-LABEL: test6:
; CHECK:       bnezc
; CHECK-NEXT:  nop

define i32 @test6(i32 %a, i32 %b) {
entry:
  %0 = icmp ugt i32 %a, %b
  br i1 %0, label %cond.true, label %cond.false
cond.true:
  %1 = call i32 @f(i32 %a)
  ret i32 %1
cond.false:
  %2 = call i32 @g()
  ret i32 %2
}

; CHECK-LABEL: test7:
; CHECK:       beqzc
; CHECK-NEXT:  nop

define i32 @test7(i32 %a, i32 %b) {
entry:
  %0 = icmp uge i32 %a, %b
  br i1 %0, label %cond.true, label %cond.false
cond.true:
  %1 = call i32 @f(i32 %a)
  ret i32 %1
cond.false:
  %2 = call i32 @g()
  ret i32 %2
}

; CHECK-LABEL: test8:
; CHECK:       bnezc
; CHECK-NEXT:  nop

define i32 @test8(i32 %a, i32 %b) {
entry:
  %0 = icmp ult i32 %a, %b
  br i1 %0, label %cond.true, label %cond.false
cond.true:
  %1 = call i32 @f(i32 %a)
  ret i32 %1
cond.false:
  %2 = call i32 @g()
  ret i32 %2
}

; CHECK-LABEL: test9:
; CHECK:       beqzc
; CHECK-NEXT:  nop

define i32 @test9(i32 %a, i32 %b) {
entry:
  %0 = icmp ule i32 %a, %b
  br i1 %0, label %cond.true, label %cond.false
cond.true:
  %1 = call i32 @f(i32 %a)
  ret i32 %1
cond.false:
  %2 = call i32 @g()
  ret i32 %2
}
