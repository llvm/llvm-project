; RUN: llc < %s -asm-verbose=0 -mtriple=arm64-eabi -mattr=+bf16 | FileCheck %s
; RUN: llc < %s -asm-verbose=0 -mtriple=aarch64-eabi -mattr=+bf16 | FileCheck %s

; test argument passing and simple load/store

define bfloat @test_load(bfloat* %p) nounwind {
; CHECK-LABEL: test_load:
; CHECK-NEXT: ldr h0, [x0]
; CHECK-NEXT: ret
  %tmp1 = load bfloat, bfloat* %p, align 16
  ret bfloat %tmp1
}

define <4 x bfloat> @test_vec_load(<4 x bfloat>* %p) nounwind {
; CHECK-LABEL: test_vec_load:
; CHECK-NEXT: ldr d0, [x0]
; CHECK-NEXT: ret
  %tmp1 = load <4 x bfloat>, <4 x bfloat>* %p, align 16
  ret <4 x bfloat> %tmp1
}

define void @test_store(bfloat* %a, bfloat %b) nounwind {
; CHECK-LABEL: test_store:
; CHECK-NEXT: str h0, [x0]
; CHECK-NEXT: ret
  store bfloat %b, bfloat* %a, align 16
  ret void
}

; Simple store of v4bf16
define void @test_vec_store(<4 x bfloat>* %a, <4 x bfloat> %b) nounwind {
; CHECK-LABEL: test_vec_store:
; CHECK-NEXT: str d0, [x0]
; CHECK-NEXT: ret
entry:
  store <4 x bfloat> %b, <4 x bfloat>* %a, align 16
  ret void
}
