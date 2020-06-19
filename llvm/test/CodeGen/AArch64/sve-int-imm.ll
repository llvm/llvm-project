; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; SVE Arith Vector Immediate Unpredicated CodeGen
;

; ADD
define <vscale x 16 x i8> @add_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: add_i8_low
; CHECK: add  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 30, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %res =  add <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @add_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: add_i16_low
; CHECK: add  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 30, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  add <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @add_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: add_i16_high
; CHECK: add  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 1024, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  add <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @add_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: add_i32_low
; CHECK: add  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 30, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res = add <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @add_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: add_i32_high
; CHECK: add  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1024, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  add <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @add_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: add_i64_low
; CHECK: add  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 30, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  add <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @add_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: add_i64_high
; CHECK: add  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 1024, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res = add <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i64> %res
}

; SUBR
define <vscale x 16 x i8> @subr_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: subr_i8_low
; CHECK: subr  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 30, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %res =  sub <vscale x 16 x i8> %splat, %a
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @subr_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: subr_i16_low
; CHECK: subr  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 30, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  sub <vscale x 8 x i16> %splat, %a
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @subr_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: subr_i16_high
; CHECK: subr  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 1024, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  sub <vscale x 8 x i16> %splat, %a
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @subr_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: subr_i32_low
; CHECK: subr  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 30, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  sub <vscale x 4 x i32> %splat, %a
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @subr_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: subr_i32_high
; CHECK: subr  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1024, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  sub <vscale x 4 x i32> %splat, %a
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @subr_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: subr_i64_low
; CHECK: subr  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 30, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  sub <vscale x 2 x i64> %splat, %a
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @subr_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: subr_i64_high
; CHECK: subr  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 1024, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  sub <vscale x 2 x i64> %splat, %a
  ret <vscale x 2 x i64> %res
}

; SUB
define <vscale x 16 x i8> @sub_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sub_i8_low
; CHECK: sub  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 30, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %res =  sub <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @sub_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sub_i16_low
; CHECK: sub  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 30, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  sub <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @sub_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sub_i16_high
; CHECK: sub  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 1024, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  sub <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sub_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sub_i32_low
; CHECK: sub  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 30, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res = sub <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @sub_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sub_i32_high
; CHECK: sub  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1024, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  sub <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sub_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sub_i64_low
; CHECK: sub  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 30, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  sub <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @sub_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sub_i64_high
; CHECK: sub  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 1024, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res = sub <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i64> %res
}

; SQADD
define <vscale x 16 x i8> @sqadd_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sqadd_i8_low
; CHECK: sqadd  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 30, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %res =  call <vscale x 16 x i8> @llvm.sadd.sat.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @sqadd_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqadd_i16_low
; CHECK: sqadd  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 30, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  call <vscale x 8 x i16> @llvm.sadd.sat.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @sqadd_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqadd_i16_high
; CHECK: sqadd  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 1024, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  call <vscale x 8 x i16> @llvm.sadd.sat.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqadd_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqadd_i32_low
; CHECK: sqadd  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 30, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  call <vscale x 4 x i32> @llvm.sadd.sat.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @sqadd_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqadd_i32_high
; CHECK: sqadd  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1024, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  call <vscale x 4 x i32> @llvm.sadd.sat.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqadd_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqadd_i64_low
; CHECK: sqadd  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 30, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  call <vscale x 2 x i64> @llvm.sadd.sat.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @sqadd_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqadd_i64_high
; CHECK: sqadd  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 1024, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  call <vscale x 2 x i64> @llvm.sadd.sat.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %res
}

; UQADD
define <vscale x 16 x i8> @uqadd_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: uqadd_i8_low
; CHECK: uqadd  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 30, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %res =  call <vscale x 16 x i8> @llvm.uadd.sat.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @uqadd_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqadd_i16_low
; CHECK: uqadd  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 30, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  call <vscale x 8 x i16> @llvm.uadd.sat.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @uqadd_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqadd_i16_high
; CHECK: uqadd  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 1024, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  call <vscale x 8 x i16> @llvm.uadd.sat.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @uqadd_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqadd_i32_low
; CHECK: uqadd  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 30, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  call <vscale x 4 x i32> @llvm.uadd.sat.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @uqadd_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqadd_i32_high
; CHECK: uqadd  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1024, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  call <vscale x 4 x i32> @llvm.uadd.sat.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @uqadd_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqadd_i64_low
; CHECK: uqadd  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 30, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  call <vscale x 2 x i64> @llvm.uadd.sat.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @uqadd_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqadd_i64_high
; CHECK: uqadd  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 1024, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  call <vscale x 2 x i64> @llvm.uadd.sat.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %res
}

; SQSUB
define <vscale x 16 x i8> @sqsub_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sqsub_i8_low
; CHECK: sqsub  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 30, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %res =  call <vscale x 16 x i8> @llvm.ssub.sat.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @sqsub_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqsub_i16_low
; CHECK: sqsub  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 30, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  call <vscale x 8 x i16> @llvm.ssub.sat.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @sqsub_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqsub_i16_high
; CHECK: sqsub  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 1024, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  call <vscale x 8 x i16> @llvm.ssub.sat.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqsub_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqsub_i32_low
; CHECK: sqsub  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 30, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  call <vscale x 4 x i32> @llvm.ssub.sat.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @sqsub_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqsub_i32_high
; CHECK: sqsub  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1024, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  call <vscale x 4 x i32> @llvm.ssub.sat.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqsub_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqsub_i64_low
; CHECK: sqsub  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 30, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  call <vscale x 2 x i64> @llvm.ssub.sat.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @sqsub_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqsub_i64_high
; CHECK: sqsub  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 1024, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  call <vscale x 2 x i64> @llvm.ssub.sat.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %res
}

; UQSUB
define <vscale x 16 x i8> @uqsub_i8_low(<vscale x 16 x i8> %a) {
; CHECK-LABEL: uqsub_i8_low
; CHECK: uqsub  z0.b, z0.b, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 30, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %res =  call <vscale x 16 x i8> @llvm.usub.sat.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @uqsub_i16_low(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqsub_i16_low
; CHECK: uqsub  z0.h, z0.h, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 30, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  call <vscale x 8 x i16> @llvm.usub.sat.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @uqsub_i16_high(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqsub_i16_high
; CHECK: uqsub  z0.h, z0.h, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 1024, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res =  call <vscale x 8 x i16> @llvm.usub.sat.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @uqsub_i32_low(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqsub_i32_low
; CHECK: uqsub  z0.s, z0.s, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 30, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  call <vscale x 4 x i32> @llvm.usub.sat.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @uqsub_i32_high(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqsub_i32_high
; CHECK: uqsub  z0.s, z0.s, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1024, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res =  call <vscale x 4 x i32> @llvm.usub.sat.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @uqsub_i64_low(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqsub_i64_low
; CHECK: uqsub  z0.d, z0.d, #30
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 30, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  call <vscale x 2 x i64> @llvm.usub.sat.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @uqsub_i64_high(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqsub_i64_high
; CHECK: uqsub  z0.d, z0.d, #1024
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 1024, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res =  call <vscale x 2 x i64> @llvm.usub.sat.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %res
}

declare <vscale x 16 x i8> @llvm.sadd.sat.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.sadd.sat.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.sadd.sat.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.sadd.sat.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.uadd.sat.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.uadd.sat.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.uadd.sat.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.uadd.sat.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.ssub.sat.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.ssub.sat.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.ssub.sat.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.ssub.sat.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.usub.sat.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.usub.sat.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.usub.sat.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.usub.sat.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
