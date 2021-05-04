; RUN: llc < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 16 x i8> @andv_zero_fill(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) #0 {
; CHECK-LABEL: andv_zero_fill:
; CHECK: andv b0, p0, z0.b
; CHECK-NEXT: ret
  %t1 = call i8 @llvm.aarch64.sve.andv.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a)
  %t2 = insertelement <vscale x 16 x i8> zeroinitializer, i8 %t1, i64 0
  ret <vscale x 16 x i8> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 8 x i16> @eorv_zero_fill(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) #0 {
; CHECK-LABEL: eorv_zero_fill:
; CHECK: eorv h0, p0, z0.h
; CHECK-NEXT: ret
  %t1 = call i16 @llvm.aarch64.sve.eorv.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a)
  %t2 = insertelement <vscale x 8 x i16> zeroinitializer, i16 %t1, i64 0
  ret <vscale x 8 x i16> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 2 x double> @fadda_zero_fill(<vscale x 2 x i1> %pg, double %init, <vscale x 2 x double> %a) #0 {
; CHECK-LABEL: fadda_zero_fill:
; CHECK: fadda d0, p0, d0, z1.d
; CHECK-NEXT: ret
  %t1 = call double @llvm.aarch64.sve.fadda.nxv2f64(<vscale x 2 x i1> %pg, double %init, <vscale x 2 x double> %a)
  %t2 = insertelement <vscale x 2 x double> zeroinitializer, double %t1, i64 0
  ret <vscale x 2 x double> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 4 x float> @faddv_zero_fill(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) #0 {
; CHECK-LABEL: faddv_zero_fill:
; CHECK: faddv s0, p0, z0.s
; CHECK-NEXT: ret
  %t1 = call float @llvm.aarch64.sve.faddv.nxv4f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a)
  %t2 = insertelement <vscale x 4 x float> zeroinitializer, float %t1, i64 0
  ret <vscale x 4 x float> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 8 x half> @fmaxv_zero_fill(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) #0 {
; CHECK-LABEL: fmaxv_zero_fill:
; CHECK: fmaxv h0, p0, z0.h
; CHECK-NEXT: ret
  %t1 = call half @llvm.aarch64.sve.fmaxv.nxv8f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a)
  %t2 = insertelement <vscale x 8 x half> zeroinitializer, half %t1, i64 0
  ret <vscale x 8 x half> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 2 x float> @fmaxnmv_zero_fill(<vscale x 2 x i1> %pg, <vscale x 2 x float> %a) #0 {
; CHECK-LABEL: fmaxnmv_zero_fill:
; CHECK: fmaxnmv s0, p0, z0.s
; CHECK-NEXT: ret
  %t1 = call float @llvm.aarch64.sve.fmaxnmv.nxv2f32(<vscale x 2 x i1> %pg, <vscale x 2 x float> %a)
  %t2 = insertelement <vscale x 2 x float> zeroinitializer, float %t1, i64 0
  ret <vscale x 2 x float> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 2 x float> @fminnmv_zero_fill(<vscale x 2 x i1> %pg, <vscale x 2 x float> %a) #0 {
; CHECK-LABEL: fminnmv_zero_fill:
; CHECK: fminnmv s0, p0, z0.s
; CHECK-NEXT: ret
  %t1 = call float @llvm.aarch64.sve.fminnmv.nxv2f32(<vscale x 2 x i1> %pg, <vscale x 2 x float> %a)
  %t2 = insertelement <vscale x 2 x float> zeroinitializer, float %t1, i64 0
  ret <vscale x 2 x float> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 2 x float> @fminv_zero_fill(<vscale x 2 x i1> %pg, <vscale x 2 x float> %a) #0 {
; CHECK-LABEL: fminv_zero_fill:
; CHECK: fminv s0, p0, z0.s
; CHECK-NEXT: ret
  %t1 = call float @llvm.aarch64.sve.fminv.nxv2f32(<vscale x 2 x i1> %pg, <vscale x 2 x float> %a)
  %t2 = insertelement <vscale x 2 x float> zeroinitializer, float %t1, i64 0
  ret <vscale x 2 x float> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 4 x i32> @orv_zero_fill(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: orv_zero_fill:
; CHECK: orv s0, p0, z0.s
; CHECK-NEXT: ret
  %t1 = call i32 @llvm.aarch64.sve.orv.nxv4i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a)
  %t2 = insertelement <vscale x 4 x i32> zeroinitializer, i32 %t1, i64 0
  ret <vscale x 4 x i32> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 2 x i64> @saddv_zero_fill(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) #0 {
; CHECK-LABEL: saddv_zero_fill:
; CHECK: saddv d0, p0, z0.b
; CHECK-NEXT: ret
  %t1 = call i64 @llvm.aarch64.sve.saddv.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a)
  %t2 = insertelement <vscale x 2 x i64> zeroinitializer, i64 %t1, i64 0
  ret <vscale x 2 x i64> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 2 x i64> @smaxv_zero_fill(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: smaxv_zero_fill:
; CHECK: smaxv d0, p0, z0.d
; CHECK-NEXT: ret
  %t1 = call i64 @llvm.aarch64.sve.smaxv.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a)
  %t2 = insertelement <vscale x 2 x i64> zeroinitializer, i64 %t1, i64 0
  ret <vscale x 2 x i64> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 4 x i32> @sminv_zero_fill(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: sminv_zero_fill:
; CHECK: sminv s0, p0, z0.s
; CHECK-NEXT: ret
  %t1 = call i32 @llvm.aarch64.sve.sminv.nxv4i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a)
  %t2 = insertelement <vscale x 4 x i32> zeroinitializer, i32 %t1, i64 0
  ret <vscale x 4 x i32> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 2 x i64> @uaddv_zero_fill(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) #0 {
; CHECK-LABEL: uaddv_zero_fill:
; CHECK: uaddv d0, p0, z0.h
; CHECK-NEXT: ret
  %t1 = call i64 @llvm.aarch64.sve.uaddv.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a)
  %t2 = insertelement <vscale x 2 x i64> zeroinitializer, i64 %t1, i64 0
  ret <vscale x 2 x i64> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 16 x i8> @umaxv_zero_fill(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) #0 {
; CHECK-LABEL: umaxv_zero_fill:
; CHECK: umaxv b0, p0, z0.b
; CHECK-NEXT: ret
  %t1 = call i8 @llvm.aarch64.sve.umaxv.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a)
  %t2 = insertelement <vscale x 16 x i8> zeroinitializer, i8 %t1, i64 0
  ret <vscale x 16 x i8> %t2
}

; Ensure we rely on the reduction's implicit zero filling.
define <vscale x 2 x i64> @uminv_zero_fill(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: uminv_zero_fill:
; CHECK: uminv d0, p0, z0.d
; CHECK-NEXT: ret
  %t1 = call i64 @llvm.aarch64.sve.uminv.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a)
  %t2 = insertelement <vscale x 2 x i64> zeroinitializer, i64 %t1, i64 0
  ret <vscale x 2 x i64> %t2
}

; Ensure explicit zeroing when inserting into a lane other than 0.
; NOTE: This test doesn't care about the exact way an insert is code generated,
; so only checks the presence of one instruction from the expected chain.
define <vscale x 2 x i64> @zero_fill_non_zero_index(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: zero_fill_non_zero_index:
; CHECK: uminv d{{[0-9]+}}, p0, z0.d
; CHECK: mov z{{[0-9]+}}.d, p{{[0-9]+}}/m, x{{[0-9]+}}
; CHECK: ret
  %t1 = call i64 @llvm.aarch64.sve.uminv.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a)
  %t2 = insertelement <vscale x 2 x i64> zeroinitializer, i64 %t1, i64 1
  ret <vscale x 2 x i64> %t2
}

; Ensure explicit zeroing when the result vector is larger than that produced by
; the reduction instruction.
define <vscale x 4 x i64> @zero_fill_type_mismatch(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: zero_fill_type_mismatch:
; CHECK: uminv d0, p0, z0.d
; CHECK-NEXT: mov z1.d, #0
; CHECK-NEXT: ret
  %t1 = call i64 @llvm.aarch64.sve.uminv.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a)
  %t2 = insertelement <vscale x 4 x i64> zeroinitializer, i64 %t1, i64 0
  ret <vscale x 4 x i64> %t2
}

; Ensure explicit zeroing when extracting an element from an operation that
; cannot guarantee lanes 1-N are zero.
; NOTE: This test doesn't care about the exact way an insert is code generated,
; so only checks the presence of one instruction from the expected chain.
define <vscale x 2 x i64> @zero_fill_no_zero_upper_lanes(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: zero_fill_no_zero_upper_lanes:
; CHECK: umin z{{[0-9]+}}.d, p0/m, z0.d, z0.d
; CHECK: mov z{{[0-9]+}}.d, p{{[0-9]+}}/m, x{{[0-9]+}}
; CHECK: ret
  %t1 = call <vscale x 2 x i64> @llvm.aarch64.sve.umin.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %a)
  %t2 = extractelement <vscale x 2 x i64> %t1, i64 0
  %t3 = insertelement <vscale x 2 x i64> zeroinitializer, i64 %t2, i64 0
  ret <vscale x 2 x i64> %t3
}

declare i8 @llvm.aarch64.sve.andv.nxv2i8(<vscale x 2 x i1>, <vscale x 2 x i8>)
declare i8 @llvm.aarch64.sve.andv.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>)

declare i8 @llvm.aarch64.sve.eorv.nxv2i8(<vscale x 2 x i1>, <vscale x 2 x i8>)
declare i16 @llvm.aarch64.sve.eorv.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>)

declare float @llvm.aarch64.sve.fadda.nxv2f32(<vscale x 2 x i1>, float, <vscale x 2 x float>)
declare double @llvm.aarch64.sve.fadda.nxv2f64(<vscale x 2 x i1>, double, <vscale x 2 x double>)

declare float @llvm.aarch64.sve.faddv.nxv2f32(<vscale x 2 x i1>, <vscale x 2 x float>)
declare float @llvm.aarch64.sve.faddv.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>)

declare float @llvm.aarch64.sve.fmaxnmv.nxv2f32(<vscale x 2 x i1>, <vscale x 2 x float>)

declare half @llvm.aarch64.sve.fmaxv.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.fmaxv.nxv2f32(<vscale x 2 x i1>, <vscale x 2 x float>)

declare float @llvm.aarch64.sve.fminv.nxv2f32(<vscale x 2 x i1>, <vscale x 2 x float>)

declare float @llvm.aarch64.sve.fminnmv.nxv2f32(<vscale x 2 x i1>, <vscale x 2 x float>)

declare i8 @llvm.aarch64.sve.orv.nxv2i8(<vscale x 2 x i1>, <vscale x 2 x i8>)
declare i32 @llvm.aarch64.sve.orv.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>)

declare i64 @llvm.aarch64.sve.saddv.nxv2i8(<vscale x 2 x i1>, <vscale x 2 x i8>)
declare i64 @llvm.aarch64.sve.saddv.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>)

declare i8 @llvm.aarch64.sve.smaxv.nxv2i8(<vscale x 2 x i1>, <vscale x 2 x i8>)
declare i64 @llvm.aarch64.sve.smaxv.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>)

declare i8 @llvm.aarch64.sve.sminv.nxv2i8(<vscale x 2 x i1>, <vscale x 2 x i8>)
declare i32 @llvm.aarch64.sve.sminv.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>)

declare i64 @llvm.aarch64.sve.uaddv.nxv2i8(<vscale x 2 x i1>, <vscale x 2 x i8>)
declare i64 @llvm.aarch64.sve.uaddv.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>)

declare i8 @llvm.aarch64.sve.umaxv.nxv2i8(<vscale x 2 x i1>, <vscale x 2 x i8>)
declare i8 @llvm.aarch64.sve.umaxv.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>)

declare i8 @llvm.aarch64.sve.uminv.nxv2i8(<vscale x 2 x i1>, <vscale x 2 x i8>)
declare i64 @llvm.aarch64.sve.uminv.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>)

declare <vscale x 2 x i64> @llvm.aarch64.sve.umin.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

attributes #0 = { "target-features"="+sve" }
