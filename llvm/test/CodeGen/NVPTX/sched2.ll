; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

define void @foo(<2 x i32>* %a) {
; CHECK: .func foo
; CHECK: ld.v2.u32
; CHECK-NEXT: ld.v2.u32
; CHECK-NEXT: ld.v2.u32
; CHECK-NEXT: ld.v2.u32
; CHECK-NEXT: add.s32
; CHECK-NEXT: add.s32
; CHECK-NEXT: add.s32
; CHECK-NEXT: add.s32
; CHECK-NEXT: add.s32
; CHECK-NEXT: add.s32
  %ptr0 = getelementptr <2 x i32>, <2 x i32>* %a, i32 0
  %val0 = load <2 x i32>, <2 x i32>* %ptr0
  %ptr1 = getelementptr <2 x i32>, <2 x i32>* %a, i32 1
  %val1 = load <2 x i32>, <2 x i32>* %ptr1
  %ptr2 = getelementptr <2 x i32>, <2 x i32>* %a, i32 2
  %val2 = load <2 x i32>, <2 x i32>* %ptr2
  %ptr3 = getelementptr <2 x i32>, <2 x i32>* %a, i32 3
  %val3 = load <2 x i32>, <2 x i32>* %ptr3

  %t0 = add <2 x i32> %val0, %val1
  %t1 = add <2 x i32> %t0, %val2
  %t2 = add <2 x i32> %t1, %val3

  store <2 x i32> %t2, <2 x i32>* %a

  ret void
}

