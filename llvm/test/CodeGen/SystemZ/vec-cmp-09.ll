; Test usage of VBLEND on arch15.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=arch15 | FileCheck %s

define <16 x i8> @f1(<16 x i8> %val1, <16 x i8> %val2, <16 x i8> %val3) {
; CHECK-LABEL: f1:
; CHECK: vblendb %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp slt <16 x i8> %val1, zeroinitializer
  %ret = select <16 x i1> %cmp, <16 x i8> %val2, <16 x i8> %val3
  ret <16 x i8> %ret
}

define <16 x i8> @f2(<16 x i8> %val1, <16 x i8> %val2, <16 x i8> %val3) {
; CHECK-LABEL: f2:
; CHECK: vblendb %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp sge <16 x i8> %val1, zeroinitializer
  %ret = select <16 x i1> %cmp, <16 x i8> %val2, <16 x i8> %val3
  ret <16 x i8> %ret
}

define <16 x i8> @f3(<16 x i8> %val1, <16 x i8> %val2, <16 x i8> %val3) {
; CHECK-LABEL: f3:
; CHECK: vblendb %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %mask = and <16 x i8> %val1, <i8 128, i8 128, i8 128, i8 128,
                                i8 128, i8 128, i8 128, i8 128,
                                i8 128, i8 128, i8 128, i8 128,
                                i8 128, i8 128, i8 128, i8 128>;
  %cmp = icmp ne <16 x i8> %mask, zeroinitializer
  %ret = select <16 x i1> %cmp, <16 x i8> %val2, <16 x i8> %val3
  ret <16 x i8> %ret
}

define <16 x i8> @f4(<16 x i8> %val1, <16 x i8> %val2, <16 x i8> %val3) {
; CHECK-LABEL: f4:
; CHECK: vblendb %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %mask = and <16 x i8> %val1, <i8 128, i8 128, i8 128, i8 128,
                                i8 128, i8 128, i8 128, i8 128,
                                i8 128, i8 128, i8 128, i8 128,
                                i8 128, i8 128, i8 128, i8 128>;
  %cmp = icmp eq <16 x i8> %mask, zeroinitializer
  %ret = select <16 x i1> %cmp, <16 x i8> %val2, <16 x i8> %val3
  ret <16 x i8> %ret
}

define <8 x i16> @f5(<8 x i16> %val1, <8 x i16> %val2, <8 x i16> %val3) {
; CHECK-LABEL: f5:
; CHECK: vblendh %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp slt <8 x i16> %val1, zeroinitializer
  %ret = select <8 x i1> %cmp, <8 x i16> %val2, <8 x i16> %val3
  ret <8 x i16> %ret
}

define <8 x i16> @f6(<8 x i16> %val1, <8 x i16> %val2, <8 x i16> %val3) {
; CHECK-LABEL: f6:
; CHECK: vblendh %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp sge <8 x i16> %val1, zeroinitializer
  %ret = select <8 x i1> %cmp, <8 x i16> %val2, <8 x i16> %val3
  ret <8 x i16> %ret
}

define <8 x i16> @f7(<8 x i16> %val1, <8 x i16> %val2, <8 x i16> %val3) {
; CHECK-LABEL: f7:
; CHECK: vblendh %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %mask = and <8 x i16> %val1, <i16 32768, i16 32768, i16 32768, i16 32768,
                                i16 32768, i16 32768, i16 32768, i16 32768>;
  %cmp = icmp ne <8 x i16> %mask, zeroinitializer
  %ret = select <8 x i1> %cmp, <8 x i16> %val2, <8 x i16> %val3
  ret <8 x i16> %ret
}

define <8 x i16> @f8(<8 x i16> %val1, <8 x i16> %val2, <8 x i16> %val3) {
; CHECK-LABEL: f8:
; CHECK: vblendh %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %mask = and <8 x i16> %val1, <i16 32768, i16 32768, i16 32768, i16 32768,
                                i16 32768, i16 32768, i16 32768, i16 32768>;
  %cmp = icmp eq <8 x i16> %mask, zeroinitializer
  %ret = select <8 x i1> %cmp, <8 x i16> %val2, <8 x i16> %val3
  ret <8 x i16> %ret
}

define <4 x i32> @f9(<4 x i32> %val1, <4 x i32> %val2, <4 x i32> %val3) {
; CHECK-LABEL: f9:
; CHECK: vblendf %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp slt <4 x i32> %val1, zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x i32> %val2, <4 x i32> %val3
  ret <4 x i32> %ret
}

define <4 x i32> @f10(<4 x i32> %val1, <4 x i32> %val2, <4 x i32> %val3) {
; CHECK-LABEL: f10:
; CHECK: vblendf %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp sge <4 x i32> %val1, zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x i32> %val2, <4 x i32> %val3
  ret <4 x i32> %ret
}

define <4 x i32> @f11(<4 x i32> %val1, <4 x i32> %val2, <4 x i32> %val3) {
; CHECK-LABEL: f11:
; CHECK: vblendf %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %mask = and <4 x i32> %val1, <i32 2147483648, i32 2147483648,
                                i32 2147483648, i32 2147483648>;
  %cmp = icmp ne <4 x i32> %mask, zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x i32> %val2, <4 x i32> %val3
  ret <4 x i32> %ret
}

define <4 x i32> @f12(<4 x i32> %val1, <4 x i32> %val2, <4 x i32> %val3) {
; CHECK-LABEL: f12:
; CHECK: vblendf %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %mask = and <4 x i32> %val1, <i32 2147483648, i32 2147483648,
                                i32 2147483648, i32 2147483648>;
  %cmp = icmp eq <4 x i32> %mask, zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x i32> %val2, <4 x i32> %val3
  ret <4 x i32> %ret
}

define <2 x i64> @f13(<2 x i64> %val1, <2 x i64> %val2, <2 x i64> %val3) {
; CHECK-LABEL: f13:
; CHECK: vblendg %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp slt <2 x i64> %val1, zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x i64> %val2, <2 x i64> %val3
  ret <2 x i64> %ret
}

define <2 x i64> @f14(<2 x i64> %val1, <2 x i64> %val2, <2 x i64> %val3) {
; CHECK-LABEL: f14:
; CHECK: vblendg %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp sge <2 x i64> %val1, zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x i64> %val2, <2 x i64> %val3
  ret <2 x i64> %ret
}

define <2 x i64> @f15(<2 x i64> %val1, <2 x i64> %val2, <2 x i64> %val3) {
; CHECK-LABEL: f15:
; CHECK: vblendg %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %mask = and <2 x i64> %val1, <i64 9223372036854775808,
                                i64 9223372036854775808>;
  %cmp = icmp ne <2 x i64> %mask, zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x i64> %val2, <2 x i64> %val3
  ret <2 x i64> %ret
}

define <2 x i64> @f16(<2 x i64> %val1, <2 x i64> %val2, <2 x i64> %val3) {
; CHECK-LABEL: f16:
; CHECK: vblendg %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %mask = and <2 x i64> %val1, <i64 9223372036854775808,
                                i64 9223372036854775808>;
  %cmp = icmp eq <2 x i64> %mask, zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x i64> %val2, <2 x i64> %val3
  ret <2 x i64> %ret
}

define <4 x float> @f17(<4 x i32> %val1, <4 x float> %val2, <4 x float> %val3) {
; CHECK-LABEL: f17:
; CHECK: vblendf %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp slt <4 x i32> %val1, zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x float> %val2, <4 x float> %val3
  ret <4 x float> %ret
}

define <4 x float> @f18(<4 x i32> %val1, <4 x float> %val2, <4 x float> %val3) {
; CHECK-LABEL: f18:
; CHECK: vblendf %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp sge <4 x i32> %val1, zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x float> %val2, <4 x float> %val3
  ret <4 x float> %ret
}

define <4 x float> @f19(<4 x i32> %val1, <4 x float> %val2, <4 x float> %val3) {
; CHECK-LABEL: f19:
; CHECK: vblendf %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %mask = and <4 x i32> %val1, <i32 2147483648, i32 2147483648,
                                i32 2147483648, i32 2147483648>;
  %cmp = icmp ne <4 x i32> %mask, zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x float> %val2, <4 x float> %val3
  ret <4 x float> %ret
}

define <4 x float> @f20(<4 x i32> %val1, <4 x float> %val2, <4 x float> %val3) {
; CHECK-LABEL: f20:
; CHECK: vblendf %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %mask = and <4 x i32> %val1, <i32 2147483648, i32 2147483648,
                                i32 2147483648, i32 2147483648>;
  %cmp = icmp eq <4 x i32> %mask, zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x float> %val2, <4 x float> %val3
  ret <4 x float> %ret
}

define <2 x double> @f21(<2 x i64> %val1, <2 x double> %val2, <2 x double> %val3) {
; CHECK-LABEL: f21:
; CHECK: vblendg %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp slt <2 x i64> %val1, zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x double> %val2, <2 x double> %val3
  ret <2 x double> %ret
}

define <2 x double> @f22(<2 x i64> %val1, <2 x double> %val2, <2 x double> %val3) {
; CHECK-LABEL: f22:
; CHECK: vblendg %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %cmp = icmp sge <2 x i64> %val1, zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x double> %val2, <2 x double> %val3
  ret <2 x double> %ret
}

define <2 x double> @f23(<2 x i64> %val1, <2 x double> %val2, <2 x double> %val3) {
; CHECK-LABEL: f23:
; CHECK: vblendg %v24, %v26, %v28, %v24
; CHECK-NEXT: br %r14
  %mask = and <2 x i64> %val1, <i64 9223372036854775808,
                                i64 9223372036854775808>;
  %cmp = icmp ne <2 x i64> %mask, zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x double> %val2, <2 x double> %val3
  ret <2 x double> %ret
}

define <2 x double> @f24(<2 x i64> %val1, <2 x double> %val2, <2 x double> %val3) {
; CHECK-LABEL: f24:
; CHECK: vblendg %v24, %v28, %v26, %v24
; CHECK-NEXT: br %r14
  %mask = and <2 x i64> %val1, <i64 9223372036854775808,
                                i64 9223372036854775808>;
  %cmp = icmp eq <2 x i64> %mask, zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x double> %val2, <2 x double> %val3
  ret <2 x double> %ret
}

