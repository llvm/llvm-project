; RUN: llc < %s -asm-verbose=0 -mtriple=arm64-eabi -mattr=+bf16 | FileCheck %s
; RUN: llc < %s -asm-verbose=0 -mtriple=aarch64-eabi -mattr=+bf16 | FileCheck %s

; test argument passing and simple load/store

define bfloat @test_load(ptr %p) nounwind {
; CHECK-LABEL: test_load:
; CHECK-NEXT: ldr h0, [x0]
; CHECK-NEXT: ret
  %tmp1 = load bfloat, ptr %p, align 16
  ret bfloat %tmp1
}

define bfloat @test_load_offset1(ptr %p) nounwind {
; CHECK-LABEL: test_load_offset1:
; CHECK-NEXT: ldur h0, [x0, #1]
; CHECK-NEXT: ret
  %g = getelementptr inbounds i8, ptr %p, i64 1
  %tmp1 = load bfloat, ptr %g, align 2
  ret bfloat %tmp1
}

define bfloat @test_load_offset2(ptr %p) nounwind {
; CHECK-LABEL: test_load_offset2:
; CHECK-NEXT: ldr h0, [x0, #2]
; CHECK-NEXT: ret
  %g = getelementptr inbounds i8, ptr %p, i64 2
  %tmp1 = load bfloat, ptr %g, align 2
  ret bfloat %tmp1
}

define <4 x bfloat> @test_vec_load(ptr %p) nounwind {
; CHECK-LABEL: test_vec_load:
; CHECK-NEXT: ldr d0, [x0]
; CHECK-NEXT: ret
  %tmp1 = load <4 x bfloat>, ptr %p, align 16
  ret <4 x bfloat> %tmp1
}

define void @test_store(ptr %a, bfloat %b) nounwind {
; CHECK-LABEL: test_store:
; CHECK-NEXT: str h0, [x0]
; CHECK-NEXT: ret
  store bfloat %b, ptr %a, align 16
  ret void
}

; Simple store of v4bf16
define void @test_vec_store(ptr %a, <4 x bfloat> %b) nounwind {
; CHECK-LABEL: test_vec_store:
; CHECK-NEXT: str d0, [x0]
; CHECK-NEXT: ret
entry:
  store <4 x bfloat> %b, ptr %a, align 16
  ret void
}

define <8 x bfloat> @test_build_vector_const() {
; CHECK-LABEL: test_build_vector_const:
; CHECK: mov [[TMP:w[0-9]+]], #16256
; CHECK: dup v0.8h, [[TMP]]
  ret  <8 x bfloat> <bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80>
}

define { bfloat, ptr } @test_store_post(bfloat %val, ptr %ptr) {
; CHECK-LABEL: test_store_post:
; CHECK: str h0, [x0], #2

  store bfloat %val, ptr %ptr
  %res.tmp = insertvalue { bfloat, ptr } undef, bfloat %val, 0

  %next = getelementptr bfloat, ptr %ptr, i32 1
  %res = insertvalue { bfloat, ptr } %res.tmp, ptr %next, 1

  ret { bfloat, ptr } %res
}

define { <4 x bfloat>, ptr } @test_store_post_v4bf16(<4 x bfloat> %val, ptr %ptr) {
; CHECK-LABEL: test_store_post_v4bf16:
; CHECK: str d0, [x0], #8

  store <4 x bfloat> %val, ptr %ptr
  %res.tmp = insertvalue { <4 x bfloat>, ptr } undef, <4 x bfloat> %val, 0

  %next = getelementptr <4 x bfloat>, ptr %ptr, i32 1
  %res = insertvalue { <4 x bfloat>, ptr } %res.tmp, ptr %next, 1

  ret { <4 x bfloat>, ptr } %res
}

define { <8 x bfloat>, ptr } @test_store_post_v8bf16(<8 x bfloat> %val, ptr %ptr) {
; CHECK-LABEL: test_store_post_v8bf16:
; CHECK: str q0, [x0], #16

  store <8 x bfloat> %val, ptr %ptr
  %res.tmp = insertvalue { <8 x bfloat>, ptr } undef, <8 x bfloat> %val, 0

  %next = getelementptr <8 x bfloat>, ptr %ptr, i32 1
  %res = insertvalue { <8 x bfloat>, ptr } %res.tmp, ptr %next, 1

  ret { <8 x bfloat>, ptr } %res
}

define bfloat @test_bitcast_halftobfloat(half %a) nounwind {
; CHECK-LABEL: test_bitcast_halftobfloat:
; CHECK-NEXT: ret
  %r = bitcast half %a to bfloat
  ret bfloat %r
}

define half @test_bitcast_bfloattohalf(bfloat %a) nounwind {
; CHECK-LABEL: test_bitcast_bfloattohalf:
; CHECK-NEXT: ret
  %r = bitcast bfloat %a to half
  ret half %r
}
