; RUN: llc -march=arm < %s | FileCheck %s
; Radar 7213850

define i32 @test(i8* %d, i32 %x, i32 %y) nounwind {
  %1 = ptrtoint i8* %d to i32
;CHECK: sub
  %2 = sub i32 %x, %1
  %3 = add nsw i32 %2, %y
  store i8 0, i8* %d, align 1
  ret i32 %3
}
