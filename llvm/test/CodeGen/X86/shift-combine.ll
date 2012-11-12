; RUN: llc -march=x86 < %s | FileCheck %s

@array = weak global [4 x i32] zeroinitializer

define i32 @test_lshr_and(i32 %x) {
; CHECK: test_lshr_and:
; CHECK-NOT: shrl
; CHECK: andl $12,
; CHECK: movl {{.*}}array{{.*}},
; CHECK: ret

entry:
  %tmp2 = lshr i32 %x, 2
  %tmp3 = and i32 %tmp2, 3
  %tmp4 = getelementptr [4 x i32]* @array, i32 0, i32 %tmp3
  %tmp5 = load i32* %tmp4, align 4
  ret i32 %tmp5
}

