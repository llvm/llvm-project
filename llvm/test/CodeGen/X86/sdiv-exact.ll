; RUN: llc -march=x86 < %s | FileCheck %s

define i32 @test1(i32 %x) {
  %div = sdiv exact i32 %x, 25
  ret i32 %div
; CHECK: test1:
; CHECK: imull	$-1030792151, 4(%esp)
; CHECK-NEXT: ret
}

define i32 @test2(i32 %x) {
  %div = sdiv exact i32 %x, 24
  ret i32 %div
; CHECK: test2:
; CHECK: sarl	$3
; CHECK-NEXT: imull	$-1431655765
; CHECK-NEXT: ret
}
