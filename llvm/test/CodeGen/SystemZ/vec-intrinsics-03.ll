; Test vector intrinsics added with z15.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

declare <16 x i8> @llvm.s390.vsld(<16 x i8>, <16 x i8>, i32)
declare <16 x i8> @llvm.s390.vsrd(<16 x i8>, <16 x i8>, i32)

declare {<16 x i8>, i32} @llvm.s390.vstrsb(<16 x i8>, <16 x i8>, <16 x i8>)
declare {<16 x i8>, i32} @llvm.s390.vstrsh(<8 x i16>, <8 x i16>, <16 x i8>)
declare {<16 x i8>, i32} @llvm.s390.vstrsf(<4 x i32>, <4 x i32>, <16 x i8>)
declare {<16 x i8>, i32} @llvm.s390.vstrszb(<16 x i8>, <16 x i8>, <16 x i8>)
declare {<16 x i8>, i32} @llvm.s390.vstrszh(<8 x i16>, <8 x i16>, <16 x i8>)
declare {<16 x i8>, i32} @llvm.s390.vstrszf(<4 x i32>, <4 x i32>, <16 x i8>)


; VSLD with the minimum useful value.
define <16 x i8> @test_vsld_1(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsld_1:
; CHECK: vsld %v24, %v24, %v26, 1
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsld(<16 x i8> %a, <16 x i8> %b, i32 1)
  ret <16 x i8> %res
}

; VSLD with the maximum value.
define <16 x i8> @test_vsld_7(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsld_7:
; CHECK: vsld %v24, %v24, %v26, 7
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsld(<16 x i8> %a, <16 x i8> %b, i32 7)
  ret <16 x i8> %res
}

; VSRD with the minimum useful value.
define <16 x i8> @test_vsrd_1(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsrd_1:
; CHECK: vsrd %v24, %v24, %v26, 1
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsrd(<16 x i8> %a, <16 x i8> %b, i32 1)
  ret <16 x i8> %res
}

; VSRD with the maximum value.
define <16 x i8> @test_vsrd_7(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsrd_7:
; CHECK: vsrd %v24, %v24, %v26, 7
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsrd(<16 x i8> %a, <16 x i8> %b, i32 7)
  ret <16 x i8> %res
}


; VSTRSB.
define <16 x i8> @test_vstrsb(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c,
                              i32 *%ccptr) {
; CHECK-LABEL: test_vstrsb:
; CHECK: vstrsb %v24, %v24, %v26, %v28, 0
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vstrsb(<16 x i8> %a, <16 x i8> %b,
                                                  <16 x i8> %c)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VSTRSH.
define <16 x i8> @test_vstrsh(<8 x i16> %a, <8 x i16> %b, <16 x i8> %c,
                              i32 *%ccptr) {
; CHECK-LABEL: test_vstrsh:
; CHECK: vstrsh %v24, %v24, %v26, %v28, 0
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vstrsh(<8 x i16> %a, <8 x i16> %b,
                                                  <16 x i8> %c)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VSTRSFS.
define <16 x i8> @test_vstrsf(<4 x i32> %a, <4 x i32> %b, <16 x i8> %c,
                              i32 *%ccptr) {
; CHECK-LABEL: test_vstrsf:
; CHECK: vstrsf %v24, %v24, %v26, %v28, 0
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vstrsf(<4 x i32> %a, <4 x i32> %b,
                                                  <16 x i8> %c)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VSTRSZB.
define <16 x i8> @test_vstrszb(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c,
                              i32 *%ccptr) {
; CHECK-LABEL: test_vstrszb:
; CHECK: vstrszb %v24, %v24, %v26, %v28
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vstrszb(<16 x i8> %a, <16 x i8> %b,
                                                   <16 x i8> %c)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VSTRSZH.
define <16 x i8> @test_vstrszh(<8 x i16> %a, <8 x i16> %b, <16 x i8> %c,
                              i32 *%ccptr) {
; CHECK-LABEL: test_vstrszh:
; CHECK: vstrszh %v24, %v24, %v26, %v28
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vstrszh(<8 x i16> %a, <8 x i16> %b,
                                                   <16 x i8> %c)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VSTRSZF.
define <16 x i8> @test_vstrszf(<4 x i32> %a, <4 x i32> %b, <16 x i8> %c,
                              i32 *%ccptr) {
; CHECK-LABEL: test_vstrszf:
; CHECK: vstrszf %v24, %v24, %v26, %v28
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vstrszf(<4 x i32> %a, <4 x i32> %b,
                                                   <16 x i8> %c)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

