; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; Ensure source scheduling is working

define void @foo(i32* %a) {
; CHECK: .func foo
; CHECK: ld.u32
; CHECK-NEXT: ld.u32
; CHECK-NEXT: add.s32
; CHECK-NEXT: ld.u32
; CHECK-NEXT: add.s32
; CHECK-NEXT: ld.u32
; CHECK-NEXT: add.s32
  %ptr0 = getelementptr i32, i32* %a, i32 0
  %val0 = load i32, i32* %ptr0
  %ptr1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %ptr1
  %ptr2 = getelementptr i32, i32* %a, i32 2
  %val2 = load i32, i32* %ptr2
  %ptr3 = getelementptr i32, i32* %a, i32 3
  %val3 = load i32, i32* %ptr3

  %t0 = add i32 %val0, %val1
  %t1 = add i32 %t0, %val2
  %t2 = add i32 %t1, %val3

  store i32 %t2, i32* %a

  ret void
}

