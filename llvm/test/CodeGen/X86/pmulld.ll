; RUN: llc < %s -mtriple=x86_64-linux -mattr=+sse4.1 -asm-verbose=0 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+sse4.1 -asm-verbose=0 | FileCheck %s -check-prefix=WIN64

define <4 x i32> @test1(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK-LABEL: test1:
; CHECK-NEXT: pmulld

; WIN64-LABEL: test1:
; WIN64-NEXT: movdqa  (%rcx), %xmm0
; WIN64-NEXT: pmulld  (%rdx), %xmm0
  %C = mul <4 x i32> %A, %B
  ret <4 x i32> %C
}

define <4 x i32> @test1a(<4 x i32> %A, ptr%Bp) nounwind {
; CHECK-LABEL: test1a:
; CHECK-NEXT: pmulld

; WIN64-LABEL: test1a:
; WIN64-NEXT: movdqa  (%rcx), %xmm0
; WIN64-NEXT: pmulld  (%rdx), %xmm0

  %B = load <4 x i32>, ptr %Bp
  %C = mul <4 x i32> %A, %B
  ret <4 x i32> %C
}
