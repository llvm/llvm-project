; RUN: llc -mtriple=x86_64-unknown-linux < %s | FileCheck %s

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

define void @jt2(ptr nest, ...) {
  ; CHECK: jt2:
  ; CHECK:      leaq g+1(%rip), %r11
  ; CHECK-NEXT: cmpq %r11, %r10
  ; CHECK-NEXT: jae .LBB0_1
  ; CHECK-NEXT: #
  ; CHECK-NEXT: jmp f0
  ; CHECK-NEXT: .LBB0_1:
  ; CHECK-NEXT: jmp f1
  musttail call void (...) @llvm.icall.branch.funnel(
      ptr %0,
      ptr @g, ptr @f0,
      ptr getelementptr (i8, ptr @g, i64 1), ptr @f1,
      ...
  )
  ret void
}

define void @jt3(ptr nest, ...) {
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
      ptr %0,
      ptr @g, ptr @f0,
      ptr getelementptr (i8, ptr @g, i64 2), ptr @f2,
      ptr getelementptr (i8, ptr @g, i64 1), ptr @f1,
      ...
  )
  ret void
}

define void @jt7(ptr nest, ...) {
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
      ptr %0,
      ptr @g, ptr @f0,
      ptr getelementptr (i8, ptr @g, i64 1), ptr @f1,
      ptr getelementptr (i8, ptr @g, i64 2), ptr @f2,
      ptr getelementptr (i8, ptr @g, i64 3), ptr @f3,
      ptr getelementptr (i8, ptr @g, i64 4), ptr @f4,
      ptr getelementptr (i8, ptr @g, i64 5), ptr @f5,
      ptr getelementptr (i8, ptr @g, i64 6), ptr @f6,
      ...
  )
  ret void
}

define void @jt10(ptr nest, ...) {
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
      ptr %0,
      ptr @g, ptr @f0,
      ptr getelementptr (i8, ptr @g, i64 1), ptr @f1,
      ptr getelementptr (i8, ptr @g, i64 2), ptr @f2,
      ptr getelementptr (i8, ptr @g, i64 3), ptr @f3,
      ptr getelementptr (i8, ptr @g, i64 4), ptr @f4,
      ptr getelementptr (i8, ptr @g, i64 5), ptr @f5,
      ptr getelementptr (i8, ptr @g, i64 6), ptr @f6,
      ptr getelementptr (i8, ptr @g, i64 7), ptr @f7,
      ptr getelementptr (i8, ptr @g, i64 8), ptr @f8,
      ptr getelementptr (i8, ptr @g, i64 9), ptr @f9,
      ...
  )
  ret void
}
