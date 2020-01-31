; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 -asm-verbose=0 < %s | FileCheck %s

;
; SHADD
;

define <vscale x 16 x i8> @shadd_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: shadd_i8:
; CHECK: shadd z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.shadd.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @shadd_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: shadd_i16:
; CHECK: shadd z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.shadd.nxv8i16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @shadd_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: shadd_i32:
; CHECK: shadd z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.shadd.nxv4i32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @shadd_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: shadd_i64:
; CHECK: shadd z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.shadd.nxv2i64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SHSUB
;

define <vscale x 16 x i8> @shsub_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: shsub_i8:
; CHECK: shsub z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.shsub.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @shsub_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: shsub_i16:
; CHECK: shsub z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.shsub.nxv8i16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @shsub_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: shsub_i32:
; CHECK: shsub z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.shsub.nxv4i32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @shsub_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: shsub_i64:
; CHECK: shsub z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.shsub.nxv2i64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SHSUBR
;

define <vscale x 16 x i8> @shsubr_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: shsubr_i8:
; CHECK: shsubr z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.shsubr.nxv16i8(<vscale x 16 x i1> %pg,
                                                                  <vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @shsubr_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: shsubr_i16:
; CHECK: shsubr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.shsubr.nxv8i16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @shsubr_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: shsubr_i32:
; CHECK: shsubr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.shsubr.nxv4i32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @shsubr_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: shsubr_i64:
; CHECK: shsubr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.shsubr.nxv2i64(<vscale x 2 x i1> %pg,
                                                                  <vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SQABS
;

define <vscale x 16 x i8> @sqabs_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqabs_i8:
; CHECK: sqabs z0.b, p0/m, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqabs.nxv16i8(<vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqabs_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqabs_i16:
; CHECK: sqabs z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqabs.nxv8i16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqabs_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqabs_i32:
; CHECK: sqabs z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqabs.nxv4i32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqabs_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqabs_i64:
; CHECK: sqabs z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqabs.nxv2i64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SQDMULH (Vector)
;

define <vscale x 16 x i8> @sqdmulh_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqdmulh_i8:
; CHECK: sqdmulh z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqdmulh.nxv16i8(<vscale x 16 x i8> %a,
                                                                   <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqdmulh_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqdmulh_i16:
; CHECK: sqdmulh z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmulh.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqdmulh_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqdmulh_i32:
; CHECK: sqdmulh z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmulh.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqdmulh_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqdmulh_i64:
; CHECK: sqdmulh z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmulh.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SQDMULH (Indexed)
;

define <vscale x 8 x i16> @sqdmulh_lane_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqdmulh_lane_i16:
; CHECK: sqdmulh z0.h, z0.h, z1.h[7]
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmulh.lane.nxv8i16(<vscale x 8 x i16> %a,
                                                                        <vscale x 8 x i16> %b,
                                                                        i32 7)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqdmulh_lane_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqdmulh_lane_i32:
; CHECK: sqdmulh z0.s, z0.s, z1.s[3]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmulh.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                        <vscale x 4 x i32> %b,
                                                                        i32 3);
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqdmulh_lane_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqdmulh_lane_i64:
; CHECK: sqdmulh z0.d, z0.d, z1.d[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmulh.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                        <vscale x 2 x i64> %b,
                                                                        i32 1)
  ret <vscale x 2 x i64> %out
}

;
; SQNEG
;

define <vscale x 16 x i8> @sqneg_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqneg_i8:
; CHECK: sqneg z0.b, p0/m, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqneg.nxv16i8(<vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqneg_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqneg_i16:
; CHECK: sqneg z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqneg.nxv8i16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqneg_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqneg_i32:
; CHECK: sqneg z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqneg.nxv4i32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqneg_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqneg_i64:
; CHECK: sqneg z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqneg.nxv2i64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SQRDMALH (Vectors)
;

define <vscale x 16 x i8> @sqrdmlah_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: sqrdmlah_i8:
; CHECK: sqrdmlah z0.b, z1.b, z2.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrdmlah.nxv16i8(<vscale x 16 x i8> %a,
                                                                    <vscale x 16 x i8> %b,
                                                                    <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqrdmlah_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqrdmlah_i16:
; CHECK: sqrdmlah z0.h, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmlah.nxv8i16(<vscale x 8 x i16> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrdmlah_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqrdmlah_i32:
; CHECK: sqrdmlah z0.s, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmlah.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqrdmlah_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: sqrdmlah_i64:
; CHECK: sqrdmlah z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmlah.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 2 x i64> %b,
                                                                    <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %out
}

;
; SQRDMALH (Indexed)
;

define <vscale x 8 x i16> @sqrdmlah_lane_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqrdmlah_lane_i16:
; CHECK: sqrdmlah z0.h, z1.h, z2.h[5]
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmlah.lane.nxv8i16(<vscale x 8 x i16> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         <vscale x 8 x i16> %c,
                                                                         i32 5)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrdmlah_lane_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqrdmlah_lane_i32:
; CHECK: sqrdmlah z0.s, z1.s, z2.s[1]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmlah.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         <vscale x 4 x i32> %c,
                                                                         i32 1);
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqrdmlah_lane_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: sqrdmlah_lane_i64:
; CHECK: sqrdmlah z0.d, z1.d, z2.d[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmlah.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 2 x i64> %b,
                                                                         <vscale x 2 x i64> %c,
                                                                         i32 1)
  ret <vscale x 2 x i64> %out
}

;
; SQRDMSLH (Vectors)
;

define <vscale x 16 x i8> @sqrdmlsh_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: sqrdmlsh_i8:
; CHECK: sqrdmlsh z0.b, z1.b, z2.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrdmlsh.nxv16i8(<vscale x 16 x i8> %a,
                                                                    <vscale x 16 x i8> %b,
                                                                    <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqrdmlsh_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqrdmlsh_i16:
; CHECK: sqrdmlsh z0.h, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmlsh.nxv8i16(<vscale x 8 x i16> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrdmlsh_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqrdmlsh_i32:
; CHECK: sqrdmlsh z0.s, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmlsh.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqrdmlsh_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: sqrdmlsh_i64:
; CHECK: sqrdmlsh z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmlsh.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 2 x i64> %b,
                                                                    <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %out
}

;
; SQRDMSLH (Indexed)
;

define <vscale x 8 x i16> @sqrdmlsh_lane_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqrdmlsh_lane_i16:
; CHECK: sqrdmlsh z0.h, z1.h, z2.h[4]
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmlsh.lane.nxv8i16(<vscale x 8 x i16> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         <vscale x 8 x i16> %c,
                                                                         i32 4)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrdmlsh_lane_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqrdmlsh_lane_i32:
; CHECK: sqrdmlsh z0.s, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmlsh.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         <vscale x 4 x i32> %c,
                                                                         i32 0);
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqrdmlsh_lane_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: sqrdmlsh_lane_i64:
; CHECK: sqrdmlsh z0.d, z1.d, z2.d[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmlsh.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 2 x i64> %b,
                                                                         <vscale x 2 x i64> %c,
                                                                         i32 1)
  ret <vscale x 2 x i64> %out
}

;
; SQRDMULH (Vectors)
;

define <vscale x 16 x i8> @sqrdmulh_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqrdmulh_i8:
; CHECK: sqrdmulh z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrdmulh.nxv16i8(<vscale x 16 x i8> %a,
                                                                    <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqrdmulh_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqrdmulh_i16:
; CHECK: sqrdmulh z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmulh.nxv8i16(<vscale x 8 x i16> %a,
                                                                    <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrdmulh_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqrdmulh_i32:
; CHECK: sqrdmulh z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmulh.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqrdmulh_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqrdmulh_i64:
; CHECK: sqrdmulh z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmulh.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SQRDMULH (Indexed)
;

define <vscale x 8 x i16> @sqrdmulh_lane_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqrdmulh_lane_i16:
; CHECK: sqrdmulh z0.h, z0.h, z1.h[6]
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmulh.lane.nxv8i16(<vscale x 8 x i16> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         i32 6)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrdmulh_lane_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqrdmulh_lane_i32:
; CHECK: sqrdmulh z0.s, z0.s, z1.s[2]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmulh.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         i32 2);
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqrdmulh_lane_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqrdmulh_lane_i64:
; CHECK: sqrdmulh z0.d, z0.d, z1.d[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmulh.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 2 x i64> %b,
                                                                         i32 1)
  ret <vscale x 2 x i64> %out
}

;
; SRHADD
;

define <vscale x 16 x i8> @srhadd_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: srhadd_i8:
; CHECK: srhadd z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.srhadd.nxv16i8(<vscale x 16 x i1> %pg,
                                                                  <vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @srhadd_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: srhadd_i16:
; CHECK: srhadd z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.srhadd.nxv8i16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @srhadd_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: srhadd_i32:
; CHECK: srhadd z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.srhadd.nxv4i32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @srhadd_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: srhadd_i64:
; CHECK: srhadd z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.srhadd.nxv2i64(<vscale x 2 x i1> %pg,
                                                                  <vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; UHADD
;

define <vscale x 16 x i8> @uhadd_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uhadd_i8:
; CHECK: uhadd z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uhadd.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uhadd_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uhadd_i16:
; CHECK: uhadd z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uhadd.nxv8i16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uhadd_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uhadd_i32:
; CHECK: uhadd z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uhadd.nxv4i32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uhadd_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uhadd_i64:
; CHECK: uhadd z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uhadd.nxv2i64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; UHSUB
;

define <vscale x 16 x i8> @uhsub_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uhsub_i8:
; CHECK: uhsub z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uhsub.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uhsub_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uhsub_i16:
; CHECK: uhsub z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uhsub.nxv8i16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uhsub_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uhsub_i32:
; CHECK: uhsub z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uhsub.nxv4i32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uhsub_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uhsub_i64:
; CHECK: uhsub z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uhsub.nxv2i64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; UHSUBR
;

define <vscale x 16 x i8> @uhsubr_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uhsubr_i8:
; CHECK: uhsubr z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uhsubr.nxv16i8(<vscale x 16 x i1> %pg,
                                                                  <vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uhsubr_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uhsubr_i16:
; CHECK: uhsubr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uhsubr.nxv8i16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uhsubr_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uhsubr_i32:
; CHECK: uhsubr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uhsubr.nxv4i32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uhsubr_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uhsubr_i64:
; CHECK: uhsubr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uhsubr.nxv2i64(<vscale x 2 x i1> %pg,
                                                                  <vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; URECPE
;

define <vscale x 4 x i32> @urecpe_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: urecpe_i32:
; CHECK: urecpe z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.urecpe.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

;
; URHADD
;

define <vscale x 16 x i8> @urhadd_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: urhadd_i8:
; CHECK: urhadd z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.urhadd.nxv16i8(<vscale x 16 x i1> %pg,
                                                                  <vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @urhadd_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: urhadd_i16:
; CHECK: urhadd z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.urhadd.nxv8i16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @urhadd_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: urhadd_i32:
; CHECK: urhadd z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.urhadd.nxv4i32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @urhadd_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: urhadd_i64:
; CHECK: urhadd z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.urhadd.nxv2i64(<vscale x 2 x i1> %pg,
                                                             <vscale x 2 x i64> %a,
                                                             <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; URSQRTE
;

define <vscale x 4 x i32> @ursqrte_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: ursqrte_i32:
; CHECK: ursqrte z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.ursqrte.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.shadd.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.shadd.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.shadd.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.shadd.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.shsub.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.shsub.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.shsub.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.shsub.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.shsubr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.shsubr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.shsubr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.shsubr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqabs.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqabs.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqabs.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqabs.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqdmulh.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmulh.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmulh.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmulh.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmulh.lane.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmulh.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmulh.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqneg.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqneg.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqneg.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqneg.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqrdmlah.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmlah.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmlah.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmlah.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmlah.lane.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmlah.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmlah.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqrdmlsh.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmlsh.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmlsh.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmlsh.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmlsh.lane.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmlsh.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmlsh.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqrdmulh.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmulh.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmulh.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmulh.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmulh.lane.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmulh.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmulh.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.srhadd.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.srhadd.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.srhadd.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.srhadd.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uhadd.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uhadd.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uhadd.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uhadd.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uhsub.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uhsub.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uhsub.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uhsub.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uhsubr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uhsubr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uhsubr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uhsubr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.urecpe.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.urhadd.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.urhadd.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.urhadd.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.urhadd.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.ursqrte.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
