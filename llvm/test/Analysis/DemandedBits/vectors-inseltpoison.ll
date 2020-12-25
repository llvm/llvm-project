; RUN: opt -S -demanded-bits -analyze -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s

; CHECK-DAG: DemandedBits: 0xff00 for   %x = or <2 x i32> %a, zeroinitializer
; CHECK-DAG: DemandedBits: 0xff00 for   %y = or <2 x i32> %b, zeroinitializer
; CHECK-DAG: DemandedBits: 0xff00 for   %z = or <2 x i32> %x, %y
; CHECK-DAG: DemandedBits: 0xff for   %u = lshr <2 x i32> %z, <i32 8, i32 8>
; CHECK-DAG: DemandedBits: 0xff for   %r = trunc <2 x i32> %u to <2 x i8>
define <2 x i8> @test_basic(<2 x i32> %a, <2 x i32> %b) {
  %x = or <2 x i32> %a, zeroinitializer
  %y = or <2 x i32> %b, zeroinitializer
  %z = or <2 x i32> %x, %y
  %u = lshr <2 x i32> %z, <i32 8, i32 8>
  %r = trunc <2 x i32> %u to <2 x i8>
  ret <2 x i8> %r
}

; Vector-specific instructions

; CHECK-DAG: DemandedBits: 0xff for   %x = or <2 x i32> %a, zeroinitializer
; CHECK-DAG: DemandedBits: 0xf0 for   %z = extractelement <2 x i32> %x, i32 1
; CHECK-DAG: DemandedBits: 0xf for   %y = extractelement <2 x i32> %x, i32 0
; CHECK-DAG: DemandedBits: 0xffffffff for   %u = and i32 %y, 15
; CHECK-DAG: DemandedBits: 0xffffffff for   %v = and i32 %z, 240
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = or i32 %u, %v
define i32 @test_extractelement(<2 x i32> %a) {
  %x = or <2 x i32> %a, zeroinitializer
  %y = extractelement <2 x i32> %x, i32 0
  %z = extractelement <2 x i32> %x, i32 1
  %u = and i32 %y, 15
  %v = and i32 %z, 240
  %r = or i32 %u, %v
  ret i32 %r
}

; CHECK-DAG: DemandedBits: 0xff for   %x = or i32 %a, 0
; CHECK-DAG: DemandedBits: 0xff for   %y = or i32 %b, 0
; CHECK-DAG: DemandedBits: 0xff for   %z = insertelement <2 x i32> poison, i32 %x, i32 0
; CHECK-DAG: DemandedBits: 0xff for   %u = insertelement <2 x i32> %z, i32 %y, i32 1
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = and <2 x i32> %u, <i32 255, i32 127>
define <2 x i32> @test_insertelement(i32 %a, i32 %b) {
  %x = or i32 %a, 0
  %y = or i32 %b, 0
  %z = insertelement <2 x i32> poison, i32 %x, i32 0
  %u = insertelement <2 x i32> %z, i32 %y, i32 1
  %r = and <2 x i32> %u, <i32 255, i32 127>
  ret <2 x i32> %r
}

; CHECK-DAG: DemandedBits: 0xff for   %x = or <2 x i32> %a, zeroinitializer
; CHECK-DAG: DemandedBits: 0xff for   %y = or <2 x i32> %b, zeroinitializer
; CHECK-DAG: DemandedBits: 0xff for   %z = shufflevector <2 x i32> %x, <2 x i32> %y, <3 x i32> <i32 0, i32 3, i32 1>
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = and <3 x i32> %z, <i32 255, i32 127, i32 0>
define <3 x i32> @test_shufflevector(<2 x i32> %a, <2 x i32> %b) {
  %x = or <2 x i32> %a, zeroinitializer
  %y = or <2 x i32> %b, zeroinitializer
  %z = shufflevector <2 x i32> %x, <2 x i32> %y, <3 x i32> <i32 0, i32 3, i32 1>
  %r = and <3 x i32> %z, <i32 255, i32 127, i32 0>
  ret <3 x i32> %r
}

; Shifts with splat shift amounts

; CHECK-DAG: DemandedBits: 0xf for   %x = or <2 x i32> %a, zeroinitializer
; CHECK-DAG: DemandedBits: 0xf0 for   %y = shl <2 x i32> %x, <i32 4, i32 4>
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = and <2 x i32> %y, <i32 240, i32 240>
define <2 x i32> @test_shl(<2 x i32> %a) {
  %x = or <2 x i32> %a, zeroinitializer
  %y = shl <2 x i32> %x, <i32 4, i32 4>
  %r = and <2 x i32> %y, <i32 240, i32 240>
  ret <2 x i32> %r
}

; CHECK-DAG: DemandedBits: 0xf00 for   %x = or <2 x i32> %a, zeroinitializer
; CHECK-DAG: DemandedBits: 0xf0 for   %y = ashr <2 x i32> %x, <i32 4, i32 4>
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = and <2 x i32> %y, <i32 240, i32 240>
define <2 x i32> @test_ashr(<2 x i32> %a) {
  %x = or <2 x i32> %a, zeroinitializer
  %y = ashr <2 x i32> %x, <i32 4, i32 4>
  %r = and <2 x i32> %y, <i32 240, i32 240>
  ret <2 x i32> %r
}

; CHECK-DAG: DemandedBits: 0xf00 for   %x = or <2 x i32> %a, zeroinitializer
; CHECK-DAG: DemandedBits: 0xf0 for   %y = lshr <2 x i32> %x, <i32 4, i32 4>
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = and <2 x i32> %y, <i32 240, i32 240>
define <2 x i32> @test_lshr(<2 x i32> %a) {
  %x = or <2 x i32> %a, zeroinitializer
  %y = lshr <2 x i32> %x, <i32 4, i32 4>
  %r = and <2 x i32> %y, <i32 240, i32 240>
  ret <2 x i32> %r
}

declare <2 x i32> @llvm.fshl.i32(<2 x i32>, <2 x i32>, <2 x i32>)
declare <2 x i32> @llvm.fshr.i32(<2 x i32>, <2 x i32>, <2 x i32>)

; CHECK-DAG: DemandedBits: 0xf for   %x = or <2 x i32> %a, zeroinitializer
; CHECK-DAG: DemandedBits: 0xf0000000 for   %y = or <2 x i32> %b, zeroinitializer
; CHECK-DAG: DemandedBits: 0xff for   %z = call <2 x i32> @llvm.fshl.v2i32(<2 x i32> %x, <2 x i32> %y, <2 x i32> <i32 4, i32 4>)
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = and <2 x i32> %z, <i32 255, i32 255>
define <2 x i32> @test_fshl(<2 x i32> %a, <2 x i32> %b) {
  %x = or <2 x i32> %a, zeroinitializer
  %y = or <2 x i32> %b, zeroinitializer
  %z = call <2 x i32> @llvm.fshl.i32(<2 x i32> %x, <2 x i32> %y, <2 x i32> <i32 4, i32 4>)
  %r = and <2 x i32> %z, <i32 255, i32 255>
  ret <2 x i32> %r
}

; CHECK-DAG: DemandedBits: 0xf for   %x = or <2 x i32> %a, zeroinitializer
; CHECK-DAG: DemandedBits: 0xf0000000 for   %y = or <2 x i32> %b, zeroinitializer
; CHECK-DAG: DemandedBits: 0xff for   %z = call <2 x i32> @llvm.fshr.v2i32(<2 x i32> %x, <2 x i32> %y, <2 x i32> <i32 28, i32 28>)
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = and <2 x i32> %z, <i32 255, i32 255>
define <2 x i32> @test_fshr(<2 x i32> %a, <2 x i32> %b) {
  %x = or <2 x i32> %a, zeroinitializer
  %y = or <2 x i32> %b, zeroinitializer
  %z = call <2 x i32> @llvm.fshr.i32(<2 x i32> %x, <2 x i32> %y, <2 x i32> <i32 28, i32 28>)
  %r = and <2 x i32> %z, <i32 255, i32 255>
  ret <2 x i32> %r
}

; FP / Int conversion. These have different input / output types.

; CHECK-DAG: DemandedBits: 0xffffffff for   %x = or <2 x i32> %a, zeroinitializer
define <2 x float> @test_uitofp(<2 x i32> %a) {
  %x = or <2 x i32> %a, zeroinitializer
  %r = uitofp <2 x i32> %x to <2 x float>
  ret <2 x float> %r
}

; CHECK-DAG: DemandedBits: 0xffffffff for   %y = fptoui <2 x float> %x to <2 x i32>
define <2 x i32> @test_fptoui(<2 x float> %a) {
  %x = fadd <2 x float> %a, <float 1.0, float 1.0>
  %y = fptoui <2 x float> %x to <2 x i32>
  %r = and <2 x i32> %y, <i32 255, i32 255>
  ret <2 x i32> %y
}
