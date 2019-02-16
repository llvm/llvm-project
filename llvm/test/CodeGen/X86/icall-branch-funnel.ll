; FIXME: Fix machine verifier issues and remove -verify-machineinstrs=0. PR39436.
; RUN: llc -mtriple=x86_64-unknown-linux -verify-machineinstrs=0 < %s | FileCheck %s

@g = external global i8

declare void @f0()
declare void @f1()
declare void @f2()
declare void @f3()
declare void @f4()
declare void @f5()
declare void @f6()
declare void @f7()
declare void @f8()
declare void @f9()

declare void @llvm.icall.branch.funnel(...)

define void @jt2(i8* nest, ...) {
  ; CHECK: jt2:
  ; CHECK:      leaq g+1(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB0_1
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f0
  ; CHECK-NEXT: .LBB0_1:
  ; CHECK-NEXT: jmp f1
  musttail call void (...) @llvm.icall.branch.funnel(
      i8* %0,
      i8* getelementptr (i8, i8* @g, i64 0), void ()* @f0,
      i8* getelementptr (i8, i8* @g, i64 1), void ()* @f1,
      ...
  )
  ret void
}

define void @jt3(i8* nest, ...) {
  ; CHECK: jt3:
  ; CHECK:      leaq g+1(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB1_1
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f0
  ; CHECK-NEXT: .LBB1_1:
  ; CHECK-NEXT: jne .LBB1_2
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f1
  ; CHECK-NEXT: .LBB1_2:
  ; CHECK-NEXT: jmp f2
  musttail call void (...) @llvm.icall.branch.funnel(
      i8* %0,
      i8* getelementptr (i8, i8* @g, i64 0), void ()* @f0,
      i8* getelementptr (i8, i8* @g, i64 2), void ()* @f2,
      i8* getelementptr (i8, i8* @g, i64 1), void ()* @f1,
      ...
  )
  ret void
}

define void @jt7(i8* nest, ...) {
  ; CHECK: jt7:
  ; CHECK:      leaq g+3(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB2_1
  ; CHECK-NEXT: #
  ; CHECK-NEXT: leaq g+1(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB2_6
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f0
  ; CHECK-NEXT: .LBB2_1:
  ; CHECK-NEXT: jne .LBB2_2
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f3
  ; CHECK-NEXT: .LBB2_6:
  ; CHECK-NEXT: jne .LBB2_7
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f1
  ; CHECK-NEXT: .LBB2_2:
  ; CHECK-NEXT: leaq g+5(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB2_3
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f4
  ; CHECK-NEXT: .LBB2_7:
  ; CHECK-NEXT: jmp f2
  ; CHECK-NEXT: .LBB2_3:
  ; CHECK-NEXT: jne .LBB2_4
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f5
  ; CHECK-NEXT: .LBB2_4:
  ; CHECK-NEXT: jmp f6
  musttail call void (...) @llvm.icall.branch.funnel(
      i8* %0,
      i8* getelementptr (i8, i8* @g, i64 0), void ()* @f0,
      i8* getelementptr (i8, i8* @g, i64 1), void ()* @f1,
      i8* getelementptr (i8, i8* @g, i64 2), void ()* @f2,
      i8* getelementptr (i8, i8* @g, i64 3), void ()* @f3,
      i8* getelementptr (i8, i8* @g, i64 4), void ()* @f4,
      i8* getelementptr (i8, i8* @g, i64 5), void ()* @f5,
      i8* getelementptr (i8, i8* @g, i64 6), void ()* @f6,
      ...
  )
  ret void
}

define void @jt10(i8* nest, ...) {
  ; CHECK: jt10:
  ; CHECK:      leaq g+5(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB3_1
  ; CHECK-NEXT: #
  ; CHECK-NEXT: leaq g+1(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB3_7
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f0
  ; CHECK-NEXT: .LBB3_1:
  ; CHECK-NEXT: jne .LBB3_2
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f5
  ; CHECK-NEXT: .LBB3_7:
  ; CHECK-NEXT: jne .LBB3_8
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f1
  ; CHECK-NEXT: .LBB3_2:
  ; CHECK-NEXT: leaq g+7(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB3_3
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f6
  ; CHECK-NEXT: .LBB3_8:
  ; CHECK-NEXT: leaq g+3(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB3_9
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f2
  ; CHECK-NEXT: .LBB3_3:
  ; CHECK-NEXT: jne .LBB3_4
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f7
  ; CHECK-NEXT: .LBB3_9:
  ; CHECK-NEXT: jne .LBB3_10
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f3
  ; CHECK-NEXT: .LBB3_4:
  ; CHECK-NEXT: leaq g+9(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB3_5
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f8
  ; CHECK-NEXT: .LBB3_10:
  ; CHECK-NEXT: jmp f4
  ; CHECK-NEXT: .LBB3_5:
  ; CHECK-NEXT: jmp f9
  musttail call void (...) @llvm.icall.branch.funnel(
      i8* %0,
      i8* getelementptr (i8, i8* @g, i64 0), void ()* @f0,
      i8* getelementptr (i8, i8* @g, i64 1), void ()* @f1,
      i8* getelementptr (i8, i8* @g, i64 2), void ()* @f2,
      i8* getelementptr (i8, i8* @g, i64 3), void ()* @f3,
      i8* getelementptr (i8, i8* @g, i64 4), void ()* @f4,
      i8* getelementptr (i8, i8* @g, i64 5), void ()* @f5,
      i8* getelementptr (i8, i8* @g, i64 6), void ()* @f6,
      i8* getelementptr (i8, i8* @g, i64 7), void ()* @f7,
      i8* getelementptr (i8, i8* @g, i64 8), void ()* @f8,
      i8* getelementptr (i8, i8* @g, i64 9), void ()* @f9,
      ...
  )
  ret void
}
