; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                             Signed Comparisons                             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;
; CMPEQ
;

define <vscale x 16 x i1> @ir_cmpeq_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ir_cmpeq_b
; CHECK: cmpeq p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = icmp eq <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @int_cmpeq_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: int_cmpeq_b
; CHECK: cmpeq p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpeq.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @wide_cmpeq_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: wide_cmpeq_b
; CHECK: cmpeq p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 4, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ir_cmpeq_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ir_cmpeq_h
; CHECK: cmpeq p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = icmp eq <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @int_cmpeq_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: int_cmpeq_h
; CHECK: cmpeq p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpeq.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @wide_cmpeq_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: wide_cmpeq_h
; CHECK: cmpeq p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 -16, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ir_cmpeq_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ir_cmpeq_s
; CHECK: cmpeq p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = icmp eq <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @int_cmpeq_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: int_cmpeq_s
; CHECK: cmpeq p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpeq.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @wide_cmpeq_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: wide_cmpeq_s
; CHECK: cmpeq p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 15, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ir_cmpeq_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: ir_cmpeq_d
; CHECK: cmpeq p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = icmp eq <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @int_cmpeq_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: int_cmpeq_d
; CHECK: cmpeq p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpeq.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i1> %out
}

;
; CMPGE
;

define <vscale x 16 x i1> @ir_cmpge_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ir_cmpge_b
; CHECK: cmpge p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = icmp sge <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @int_cmpge_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: int_cmpge_b
; CHECK: cmpge p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @wide_cmpge_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: wide_cmpge_b
; CHECK: cmpge p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 4, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ir_cmpge_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ir_cmpge_h
; CHECK: cmpge p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = icmp sge <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @int_cmpge_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: int_cmpge_h
; CHECK: cmpge p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @wide_cmpge_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: wide_cmpge_h
; CHECK: cmpge p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 -16, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ir_cmpge_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ir_cmpge_s
; CHECK: cmpge p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = icmp sge <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @int_cmpge_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: int_cmpge_s
; CHECK: cmpge p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @wide_cmpge_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: wide_cmpge_s
; CHECK: cmpge p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 15, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ir_cmpge_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: ir_cmpge_d
; CHECK: cmpge p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = icmp sge <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @int_cmpge_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: int_cmpge_d
; CHECK: cmpge p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpge.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i1> %out
}

;
; CMPGT
;

define <vscale x 16 x i1> @ir_cmpgt_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ir_cmpgt_b
; CHECK: cmpgt p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = icmp sgt <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @int_cmpgt_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: int_cmpgt_b
; CHECK: cmpgt p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpgt.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @wide_cmpgt_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: wide_cmpgt_b
; CHECK: cmpgt p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 4, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ir_cmpgt_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ir_cmpgt_h
; CHECK: cmpgt p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = icmp sgt <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @int_cmpgt_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: int_cmpgt_h
; CHECK: cmpgt p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpgt.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @wide_cmpgt_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: wide_cmpgt_h
; CHECK: cmpgt p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 -16, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ir_cmpgt_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ir_cmpgt_s
; CHECK: cmpgt p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = icmp sgt <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @int_cmpgt_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: int_cmpgt_s
; CHECK: cmpgt p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpgt.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @wide_cmpgt_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: wide_cmpgt_s
; CHECK: cmpgt p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 15, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ir_cmpgt_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: ir_cmpgt_d
; CHECK: cmpgt p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = icmp sgt <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @int_cmpgt_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: int_cmpgt_d
; CHECK: cmpgt p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpgt.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i1> %out
}

;
; CMPLE
;

define <vscale x 16 x i1> @ir_cmple_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ir_cmple_b
; CHECK: cmple p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = icmp sle <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @int_cmple_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: int_cmple_b
; CHECK: cmple p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %splat,
                                                                 <vscale x 16 x i8> %a)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @wide_cmple_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: wide_cmple_b
; CHECK: cmple p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 4, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmple.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ir_cmple_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ir_cmple_h
; CHECK: cmple p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = icmp sle <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @int_cmple_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: int_cmple_h
; CHECK: cmple p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %splat,
                                                                <vscale x 8 x i16> %a)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @wide_cmple_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: wide_cmple_h
; CHECK: cmple p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 -16, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmple.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ir_cmple_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ir_cmple_s
; CHECK: cmple p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = icmp sle <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @int_cmple_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: int_cmple_s
; CHECK: cmple p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %splat,
                                                                <vscale x 4 x i32> %a)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @wide_cmple_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: wide_cmple_s
; CHECK: cmple p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 15, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmple.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ir_cmple_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: ir_cmple_d
; CHECK: cmple p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = icmp sle <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @int_cmple_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: int_cmple_d
; CHECK: cmple p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpge.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %splat,
                                                                <vscale x 2 x i64> %a)
  ret <vscale x 2 x i1> %out
}

;
; CMPLT
;

define <vscale x 16 x i1> @ir_cmplt_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ir_cmplt_b
; CHECK: cmplt p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = icmp slt <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @int_cmplt_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: int_cmplt_b
; CHECK: cmplt p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpgt.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %splat,
                                                                 <vscale x 16 x i8> %a)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @wide_cmplt_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: wide_cmplt_b
; CHECK: cmplt p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 4, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmplt.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ir_cmplt_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ir_cmplt_h
; CHECK: cmplt p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = icmp slt <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @int_cmplt_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: int_cmplt_h
; CHECK: cmplt p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpgt.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %splat,
                                                                <vscale x 8 x i16> %a)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @wide_cmplt_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: wide_cmplt_h
; CHECK: cmplt p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 -16, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmplt.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ir_cmplt_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ir_cmplt_s
; CHECK: cmplt p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = icmp slt <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @int_cmplt_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: int_cmplt_s
; CHECK: cmplt p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpgt.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %splat,
                                                                <vscale x 4 x i32> %a)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @wide_cmplt_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: wide_cmplt_s
; CHECK: cmplt p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 15, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmplt.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ir_cmplt_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: ir_cmplt_d
; CHECK: cmplt p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = icmp slt <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @int_cmplt_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: int_cmplt_d
; CHECK: cmplt p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpgt.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %splat,
                                                                <vscale x 2 x i64> %a)
  ret <vscale x 2 x i1> %out
}

;
; CMPNE
;

define <vscale x 16 x i1> @ir_cmpne_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ir_cmpne_b
; CHECK: cmpne p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = icmp ne <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @int_cmpne_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: int_cmpne_b
; CHECK: cmpne p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @wide_cmpne_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: wide_cmpne_b
; CHECK: cmpne p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 4, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ir_cmpne_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ir_cmpne_h
; CHECK: cmpne p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = icmp ne <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @int_cmpne_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: int_cmpne_h
; CHECK: cmpne p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 -16, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @wide_cmpne_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: wide_cmpne_h
; CHECK: cmpne p0.h, p0/z, z0.h, #-16
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 -16, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ir_cmpne_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ir_cmpne_s
; CHECK: cmpne p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = icmp ne <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @int_cmpne_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: int_cmpne_s
; CHECK: cmpne p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 15, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @wide_cmpne_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: wide_cmpne_s
; CHECK: cmpne p0.s, p0/z, z0.s, #15
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 15, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ir_cmpne_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: ir_cmpne_d
; CHECK: cmpne p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = icmp ne <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @int_cmpne_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: int_cmpne_d
; CHECK: cmpne p0.d, p0/z, z0.d, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i1> %out
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                            Unsigned Comparisons                            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;
; CMPHI
;

define <vscale x 16 x i1> @ir_cmphi_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ir_cmphi_b
; CHECK: cmphi p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = icmp ugt <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @int_cmphi_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: int_cmphi_b
; CHECK: cmphi p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphi.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @wide_cmphi_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: wide_cmphi_b
; CHECK: cmphi p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 4, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphi.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ir_cmphi_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ir_cmphi_h
; CHECK: cmphi p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 0, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = icmp ugt <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @int_cmphi_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: int_cmphi_h
; CHECK: cmphi p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 0, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphi.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @wide_cmphi_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: wide_cmphi_h
; CHECK: cmphi p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphi.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ir_cmphi_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ir_cmphi_s
; CHECK: cmphi p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 68, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = icmp ugt <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @int_cmphi_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: int_cmphi_s
; CHECK: cmphi p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 68, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphi.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @wide_cmphi_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: wide_cmphi_s
; CHECK: cmphi p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 68, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphi.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ir_cmphi_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: ir_cmphi_d
; CHECK: cmphi p0.d, p0/z, z0.d, #127
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 127, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = icmp ugt <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @int_cmphi_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: int_cmphi_d
; CHECK: cmphi p0.d, p0/z, z0.d, #127
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 127, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmphi.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i1> %out
}

;
; CMPHS
;

define <vscale x 16 x i1> @ir_cmphs_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ir_cmphs_b
; CHECK: cmphs p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = icmp uge <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @int_cmphs_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: int_cmphs_b
; CHECK: cmphs p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @wide_cmphs_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: wide_cmphs_b
; CHECK: cmphs p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 4, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ir_cmphs_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ir_cmphs_h
; CHECK: cmphs p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 0, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = icmp uge <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @int_cmphs_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: int_cmphs_h
; CHECK: cmphs p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 0, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @wide_cmphs_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: wide_cmphs_h
; CHECK: cmphs p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ir_cmphs_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ir_cmphs_s
; CHECK: cmphs p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 68, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = icmp uge <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @int_cmphs_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: int_cmphs_s
; CHECK: cmphs p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 68, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @wide_cmphs_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: wide_cmphs_s
; CHECK: cmphs p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 68, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ir_cmphs_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: ir_cmphs_d
; CHECK: cmphs p0.d, p0/z, z0.d, #127
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 127, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = icmp uge <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @int_cmphs_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: int_cmphs_d
; CHECK: cmphs p0.d, p0/z, z0.d, #127
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 127, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmphs.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %splat)
  ret <vscale x 2 x i1> %out
}

;
; CMPLO
;

define <vscale x 16 x i1> @ir_cmplo_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ir_cmplo_b
; CHECK: cmplo p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = icmp ult <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @int_cmplo_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: int_cmplo_b
; CHECK: cmplo p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphi.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %splat,
                                                                 <vscale x 16 x i8> %a)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @wide_cmplo_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: wide_cmplo_b
; CHECK: cmplo p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 4, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmplo.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ir_cmplo_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ir_cmplo_h
; CHECK: cmplo p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 0, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = icmp ult <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @int_cmplo_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: int_cmplo_h
; CHECK: cmplo p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 0, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphi.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %splat,
                                                                <vscale x 8 x i16> %a)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @wide_cmplo_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: wide_cmplo_h
; CHECK: cmplo p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmplo.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ir_cmplo_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ir_cmplo_s
; CHECK: cmplo p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 68, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = icmp ult <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @int_cmplo_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: int_cmplo_s
; CHECK: cmplo p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 68, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphi.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %splat,
                                                                <vscale x 4 x i32> %a)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @wide_cmplo_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: wide_cmplo_s
; CHECK: cmplo p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 68, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmplo.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ir_cmplo_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: ir_cmplo_d
; CHECK: cmplo p0.d, p0/z, z0.d, #127
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 127, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = icmp ult <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @int_cmplo_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: int_cmplo_d
; CHECK: cmplo p0.d, p0/z, z0.d, #127
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 127, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmphi.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %splat,
                                                                <vscale x 2 x i64> %a)
  ret <vscale x 2 x i1> %out
}

;
; CMPLS
;

define <vscale x 16 x i1> @ir_cmpls_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ir_cmpls_b
; CHECK: cmpls p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = icmp ule <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @int_cmpls_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: int_cmpls_b
; CHECK: cmpls p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 16 x i8> undef, i8 4, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %splat,
                                                                 <vscale x 16 x i8> %a)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @wide_cmpls_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: wide_cmpls_b
; CHECK: cmpls p0.b, p0/z, z0.b, #4
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 4, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpls.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ir_cmpls_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ir_cmpls_h
; CHECK: cmpls p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 0, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = icmp ule <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @int_cmpls_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: int_cmpls_h
; CHECK: cmpls p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 8 x i16> undef, i16 0, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %splat,
                                                                <vscale x 8 x i16> %a)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @wide_cmpls_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: wide_cmpls_h
; CHECK: cmpls p0.h, p0/z, z0.h, #0
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 0, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpls.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ir_cmpls_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ir_cmpls_s
; CHECK: cmpls p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 68, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = icmp ule <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @int_cmpls_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: int_cmpls_s
; CHECK: cmpls p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 4 x i32> undef, i32 68, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %splat,
                                                                <vscale x 4 x i32> %a)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @wide_cmpls_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: wide_cmpls_s
; CHECK: cmpls p0.s, p0/z, z0.s, #68
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 68, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpls.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ir_cmpls_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: ir_cmpls_d
; CHECK: cmpls p0.d, p0/z, z0.d, #127
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 127, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = icmp ule <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @int_cmpls_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: int_cmpls_d
; CHECK: cmpls p0.d, p0/z, z0.d, #127
; CHECK-NEXT: ret
  %elt   = insertelement <vscale x 2 x i64> undef, i64 127, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmphs.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %splat,
                                                                <vscale x 2 x i64> %a)
  ret <vscale x 2 x i1> %out
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpeq.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpeq.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpeq.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmpeq.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmpge.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpgt.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpgt.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpgt.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmpgt.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmphi.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmphi.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmphi.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmphi.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmphi.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmphi.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmphi.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmphs.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmple.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmple.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmple.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmplo.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmplo.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmplo.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpls.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpls.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpls.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmplt.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmplt.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmplt.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)
