; RUN: llc -mtriple=i686-- -mattr=+sse2 < %s | FileCheck %s

define i32 @test1(i32 %x) {
  %div = sdiv exact i32 %x, 25
  ret i32 %div
; CHECK-LABEL: test1:
; CHECK: imull	$-1030792151, 4(%esp)
; CHECK-NEXT: ret
}

define i32 @test2(i32 %x) {
  %div = sdiv exact i32 %x, 24
  ret i32 %div
; CHECK-LABEL: test2:
; CHECK: sarl	$3
; CHECK-NEXT: imull	$-1431655765
; CHECK-NEXT: ret
}

define <4 x i32> @test3(<4 x i32> %x) {
  %div = sdiv exact <4 x i32> %x, <i32 24, i32 24, i32 24, i32 24>
  ret <4 x i32> %div
; CHECK-LABEL: test3:
; CHECK: psrad	$3,
; CHECK: pmuludq
; CHECK: pmuludq
; CHECK-NOT: psrad
; CHECK: ret
}
