; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; SMAX

define <vscale x 16 x i8> @smax_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: smax_i8:
; CHECK: smax z0.b, z0.b, #-128
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %elt = insertelement <vscale x 16 x i8> undef, i8 -128, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.smax.nxv16i8(<vscale x 16 x i1> %pg,
                                                                <vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @smax_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: smax_i16:
; CHECK: smax z0.h, z0.h, #127
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %elt = insertelement <vscale x 8 x i16> undef, i16 127, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.smax.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @smax_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: smax_i32:
; CHECK: smax z0.s, z0.s, #-128
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %elt = insertelement <vscale x 4 x i32> undef, i32 -128, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.smax.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @smax_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: smax_i64:
; CHECK: smax z0.d, z0.d, #127
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %elt = insertelement <vscale x 2 x i64> undef, i64 127, i64 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.smax.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

; SMIN

define <vscale x 16 x i8> @smin_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: smin_i8:
; CHECK: smin z0.b, z0.b, #127
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %elt = insertelement <vscale x 16 x i8> undef, i8 127, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.smin.nxv16i8(<vscale x 16 x i1> %pg,
                                                                <vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @smin_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: smin_i16:
; CHECK: smin z0.h, z0.h, #-128
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %elt = insertelement <vscale x 8 x i16> undef, i16 -128, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.smin.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @smin_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: smin_i32:
; CHECK: smin z0.s, z0.s, #127
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %elt = insertelement <vscale x 4 x i32> undef, i32 127, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.smin.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @smin_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: smin_i64:
; CHECK: smin z0.d, z0.d, #-128
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %elt = insertelement <vscale x 2 x i64> undef, i64 -128, i64 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.smin.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

; UMAX

define <vscale x 16 x i8> @umax_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: umax_i8:
; CHECK: umax z0.b, z0.b, #0
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %elt = insertelement <vscale x 16 x i8> undef, i8 0, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.umax.nxv16i8(<vscale x 16 x i1> %pg,
                                                                <vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @umax_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: umax_i16:
; CHECK: umax z0.h, z0.h, #255
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %elt = insertelement <vscale x 8 x i16> undef, i16 255, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.umax.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @umax_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: umax_i32:
; CHECK: umax z0.s, z0.s, #0
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %elt = insertelement <vscale x 4 x i32> undef, i32 0, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.umax.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @umax_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: umax_i64:
; CHECK: umax z0.d, z0.d, #255
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %elt = insertelement <vscale x 2 x i64> undef, i64 255, i64 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.umax.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

; UMIN

define <vscale x 16 x i8> @umin_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: umin_i8:
; CHECK: umin z0.b, z0.b, #255
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %elt = insertelement <vscale x 16 x i8> undef, i8 255, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.umin.nxv16i8(<vscale x 16 x i1> %pg,
                                                                <vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @umin_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: umin_i16:
; CHECK: umin z0.h, z0.h, #0
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %elt = insertelement <vscale x 8 x i16> undef, i16 0, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.umin.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @umin_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: umin_i32:
; CHECK: umin z0.s, z0.s, #255
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %elt = insertelement <vscale x 4 x i32> undef, i32 255, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.umin.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @umin_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: umin_i64:
; CHECK: umin z0.d, z0.d, #0
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %elt = insertelement <vscale x 2 x i64> undef, i64 0, i64 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.umin.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

; SQADD

define <vscale x 16 x i8> @sqadd_b_lowimm(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sqadd_b_lowimm:
; CHECK: sqadd z0.b, z0.b, #27
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 27, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqadd.x.nxv16i8(<vscale x 16 x i8> %a,
                                                                   <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqadd_h_lowimm(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqadd_h_lowimm:
; CHECK: sqadd z0.h, z0.h, #43
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 43, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqadd.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 8 x i16> @sqadd_h_highimm(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqadd_h_highimm:
; CHECK: sqadd z0.h, z0.h, #2048
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 2048, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqadd.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqadd_s_lowimm(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqadd_s_lowimm:
; CHECK: sqadd z0.s, z0.s, #1
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqadd.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 4 x i32> @sqadd_s_highimm(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqadd_s_highimm:
; CHECK: sqadd z0.s, z0.s, #8192
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 8192, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqadd.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqadd_d_lowimm(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqadd_d_lowimm:
; CHECK: sqadd z0.d, z0.d, #255
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 255, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqadd.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

define <vscale x 2 x i64> @sqadd_d_highimm(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqadd_d_highimm:
; CHECK: sqadd z0.d, z0.d, #65280
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 65280, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqadd.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

; SQSUB

define <vscale x 16 x i8> @sqsub_b_lowimm(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sqsub_b_lowimm:
; CHECK: sqsub z0.b, z0.b, #27
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 27, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqsub.x.nxv16i8(<vscale x 16 x i8> %a,
                                                                   <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqsub_h_lowimm(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqsub_h_lowimm:
; CHECK: sqsub z0.h, z0.h, #43
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 43, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqsub.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 8 x i16> @sqsub_h_highimm(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqsub_h_highimm:
; CHECK: sqsub z0.h, z0.h, #2048
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 2048, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqsub.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqsub_s_lowimm(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqsub_s_lowimm:
; CHECK: sqsub z0.s, z0.s, #1
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqsub.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 4 x i32> @sqsub_s_highimm(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqsub_s_highimm:
; CHECK: sqsub z0.s, z0.s, #8192
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 8192, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqsub.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqsub_d_lowimm(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqsub_d_lowimm:
; CHECK: sqsub z0.d, z0.d, #255
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 255, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqsub.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

define <vscale x 2 x i64> @sqsub_d_highimm(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqsub_d_highimm:
; CHECK: sqsub z0.d, z0.d, #65280
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 65280, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqsub.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

; UQADD

define <vscale x 16 x i8> @uqadd_b_lowimm(<vscale x 16 x i8> %a) {
; CHECK-LABEL: uqadd_b_lowimm:
; CHECK: uqadd z0.b, z0.b, #27
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 27, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uqadd.x.nxv16i8(<vscale x 16 x i8> %a,
                                                                   <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uqadd_h_lowimm(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqadd_h_lowimm:
; CHECK: uqadd z0.h, z0.h, #43
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 43, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqadd.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 8 x i16> @uqadd_h_highimm(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqadd_h_highimm:
; CHECK: uqadd z0.h, z0.h, #2048
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 2048, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqadd.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uqadd_s_lowimm(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqadd_s_lowimm:
; CHECK: uqadd z0.s, z0.s, #1
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqadd.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

; UQSUB

define <vscale x 16 x i8> @uqsub_b_lowimm(<vscale x 16 x i8> %a) {
; CHECK-LABEL: uqsub_b_lowimm:
; CHECK: uqsub z0.b, z0.b, #27
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 27, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uqsub.x.nxv16i8(<vscale x 16 x i8> %a,
                                                                   <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uqsub_h_lowimm(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqsub_h_lowimm:
; CHECK: uqsub z0.h, z0.h, #43
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 43, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqsub.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 8 x i16> @uqsub_h_highimm(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqsub_h_highimm:
; CHECK: uqsub z0.h, z0.h, #2048
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 2048, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqsub.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uqsub_s_lowimm(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqsub_s_lowimm:
; CHECK: uqsub z0.s, z0.s, #1
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 1, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqsub.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 4 x i32> @uqsub_s_highimm(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqsub_s_highimm:
; CHECK: uqsub z0.s, z0.s, #8192
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 8192, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqsub.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uqsub_d_lowimm(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqsub_d_lowimm:
; CHECK: uqsub z0.d, z0.d, #255
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 255, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uqsub.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

define <vscale x 2 x i64> @uqsub_d_highimm(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqsub_d_highimm:
; CHECK: uqsub z0.d, z0.d, #65280
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 65280, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uqsub.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}


define <vscale x 4 x i32> @uqadd_s_highimm(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqadd_s_highimm:
; CHECK: uqadd z0.s, z0.s, #8192
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 8192, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqadd.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uqadd_d_lowimm(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqadd_d_lowimm:
; CHECK: uqadd z0.d, z0.d, #255
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 255, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uqadd.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

define <vscale x 2 x i64> @uqadd_d_highimm(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqadd_d_highimm:
; CHECK: uqadd z0.d, z0.d, #65280
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 65280, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uqadd.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

; ASR

define <vscale x 16 x i8> @asr_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: asr_i8:
; CHECK: asr z0.b, z0.b, #8
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %elt = insertelement <vscale x 16 x i8> undef, i8 8, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @asr_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: asr_i16:
; CHECK: asr z0.h, z0.h, #16
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %elt = insertelement <vscale x 8 x i16> undef, i16 16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @asr_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: asr_i32:
; CHECK: asr z0.s, z0.s, #32
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %elt = insertelement <vscale x 4 x i32> undef, i32 32, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @asr_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: asr_i64:
; CHECK: asr z0.d, z0.d, #64
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %elt = insertelement <vscale x 2 x i64> undef, i64 64, i64 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

; LSL

define <vscale x 16 x i8> @lsl_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: lsl_i8:
; CHECK: lsl z0.b, z0.b, #7
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %elt = insertelement <vscale x 16 x i8> undef, i8 7, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsl.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsl_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: lsl_i16:
; CHECK: lsl z0.h, z0.h, #15
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %elt = insertelement <vscale x 8 x i16> undef, i16 15, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsl.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsl_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: lsl_i32:
; CHECK: lsl z0.s, z0.s, #31
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %elt = insertelement <vscale x 4 x i32> undef, i32 31, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsl.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @lsl_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: lsl_i64:
; CHECK: lsl z0.d, z0.d, #63
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %elt = insertelement <vscale x 2 x i64> undef, i64 63, i64 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.lsl.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

; LSR

define <vscale x 16 x i8> @lsr_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: lsr_i8:
; CHECK: lsr z0.b, z0.b, #8
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %elt = insertelement <vscale x 16 x i8> undef, i8 8, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsr.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsr_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: lsr_i16:
; CHECK: lsr z0.h, z0.h, #16
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %elt = insertelement <vscale x 8 x i16> undef, i16 16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsr.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsr_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: lsr_i32:
; CHECK: lsr z0.s, z0.s, #32
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %elt = insertelement <vscale x 4 x i32> undef, i32 32, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsr.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @lsr_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: lsr_i64:
; CHECK: lsr z0.d, z0.d, #64
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %elt = insertelement <vscale x 2 x i64> undef, i64 64, i64 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.lsr.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqadd.x.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqadd.x.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqadd.x.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqadd.x.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqsub.x.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqsub.x.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqsub.x.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqsub.x.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uqadd.x.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqadd.x.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqadd.x.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uqadd.x.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uqsub.x.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqsub.x.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqsub.x.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uqsub.x.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.smax.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.smax.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smax.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smax.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.smin.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.smin.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smin.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smin.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.umax.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.umax.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umax.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umax.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.umin.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.umin.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umin.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umin.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsl.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsl.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsl.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.lsl.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.lsr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 %pattern)
declare <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 %pattern)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 %pattern)
declare <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 %pattern)
