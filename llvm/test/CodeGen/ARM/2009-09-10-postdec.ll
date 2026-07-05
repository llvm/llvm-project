; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s
; Radar 7213850

define i32 @test(ptr %d, i32 %x, i32 %y) nounwind {
  %1 = ptrtoint ptr %d to i32
;CHECK: sub
  %2 = sub i32 %x, %1
  %3 = add nsw i32 %2, %y
  store i8 0, ptr %d, align 1
  ret i32 %3
}
