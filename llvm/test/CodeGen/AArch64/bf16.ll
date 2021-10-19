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

define <8 x bfloat> @test_build_vector_const() {
; CHECK-LABEL: test_build_vector_const:
; CHECK: mov [[TMP:w[0-9]+]], #16256
; CHECK: dup v0.8h, [[TMP]]
  ret  <8 x bfloat> <bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80>
}

define { bfloat, bfloat* } @test_store_post(bfloat %val, bfloat* %ptr) {
; CHECK-LABEL: test_store_post:
; CHECK: str h0, [x0], #2

  store bfloat %val, bfloat* %ptr
  %res.tmp = insertvalue { bfloat, bfloat* } undef, bfloat %val, 0

  %next = getelementptr bfloat, bfloat* %ptr, i32 1
  %res = insertvalue { bfloat, bfloat* } %res.tmp, bfloat* %next, 1

  ret { bfloat, bfloat* } %res
}

define { <4 x bfloat>, <4 x bfloat>* } @test_store_post_v4bf16(<4 x bfloat> %val, <4 x bfloat>* %ptr) {
; CHECK-LABEL: test_store_post_v4bf16:
; CHECK: str d0, [x0], #8

  store <4 x bfloat> %val, <4 x bfloat>* %ptr
  %res.tmp = insertvalue { <4 x bfloat>, <4 x bfloat>* } undef, <4 x bfloat> %val, 0

  %next = getelementptr <4 x bfloat>, <4 x bfloat>* %ptr, i32 1
  %res = insertvalue { <4 x bfloat>, <4 x bfloat>* } %res.tmp, <4 x bfloat>* %next, 1

  ret { <4 x bfloat>, <4 x bfloat>* } %res
}

define { <8 x bfloat>, <8 x bfloat>* } @test_store_post_v8bf16(<8 x bfloat> %val, <8 x bfloat>* %ptr) {
; CHECK-LABEL: test_store_post_v8bf16:
; CHECK: str q0, [x0], #16

  store <8 x bfloat> %val, <8 x bfloat>* %ptr
  %res.tmp = insertvalue { <8 x bfloat>, <8 x bfloat>* } undef, <8 x bfloat> %val, 0

  %next = getelementptr <8 x bfloat>, <8 x bfloat>* %ptr, i32 1
  %res = insertvalue { <8 x bfloat>, <8 x bfloat>* } %res.tmp, <8 x bfloat>* %next, 1

  ret { <8 x bfloat>, <8 x bfloat>* } %res
}
