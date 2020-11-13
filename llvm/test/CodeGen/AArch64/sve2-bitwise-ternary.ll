; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; EOR3 (vector, bitwise, unpredicated)
;
define <vscale x 16 x i8> @eor3_i8(<vscale x 16 x i8> %a,
                                   <vscale x 16 x i8> %b,
                                   <vscale x 16 x i8> %c) {
; CHECK-LABEL: eor3_i8
; CHECK: eor3 z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.eor3.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @eor3_i16(<vscale x 8 x i16> %a,
                                    <vscale x 8 x i16> %b,
                                    <vscale x 8 x i16> %c) {
; CHECK-LABEL: eor3_i16
; CHECK: eor3 z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.eor3.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @eor3_i32(<vscale x 4 x i32> %a,
                                    <vscale x 4 x i32> %b,
                                    <vscale x 4 x i32> %c) {
; CHECK-LABEL: eor3_i32
; CHECK: eor3 z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.eor3.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @eor3_i64(<vscale x 2 x i64> %a,
                                    <vscale x 2 x i64> %b,
                                    <vscale x 2 x i64> %c) {
; CHECK-LABEL: eor3_i64
; CHECK: eor3 z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.eor3.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %res
}

;
; BCAX (vector, bitwise, unpredicated)
;
define <vscale x 16 x i8> @bcax_i8(<vscale x 16 x i8> %a,
                                   <vscale x 16 x i8> %b,
                                   <vscale x 16 x i8> %c) {
; CHECK-LABEL: bcax_i8
; CHECK: bcax z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.bcax.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @bcax_i16(<vscale x 8 x i16> %a,
                                    <vscale x 8 x i16> %b,
                                    <vscale x 8 x i16> %c) {
; CHECK-LABEL: bcax_i16
; CHECK: bcax z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.bcax.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @bcax_i32(<vscale x 4 x i32> %a,
                                    <vscale x 4 x i32> %b,
                                    <vscale x 4 x i32> %c) {
; CHECK-LABEL: bcax_i32
; CHECK: bcax z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.bcax.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @bcax_i64(<vscale x 2 x i64> %a,
                                    <vscale x 2 x i64> %b,
                                    <vscale x 2 x i64> %c) {
; CHECK-LABEL: bcax_i64
; CHECK: bcax z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.bcax.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %res
}

;
; BSL (vector, bitwise, unpredicated)
;
define <vscale x 16 x i8> @bsl_i8(<vscale x 16 x i8> %a,
                                  <vscale x 16 x i8> %b,
                                  <vscale x 16 x i8> %c) {
; CHECK-LABEL: bsl_i8
; CHECK: bsl z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.bsl.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @bsl_i16(<vscale x 8 x i16> %a,
                                   <vscale x 8 x i16> %b,
                                   <vscale x 8 x i16> %c) {
; CHECK-LABEL: bsl_i16
; CHECK: bsl z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.bsl.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @bsl_i32(<vscale x 4 x i32> %a,
                                   <vscale x 4 x i32> %b,
                                   <vscale x 4 x i32> %c) {
; CHECK-LABEL: bsl_i32
; CHECK: bsl z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.bsl.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @bsl_i64(<vscale x 2 x i64> %a,
                                   <vscale x 2 x i64> %b,
                                   <vscale x 2 x i64> %c) {
; CHECK-LABEL: bsl_i64
; CHECK: bsl z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.bsl.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %res
}

;
; BSL1N (vector, bitwise, unpredicated)
;
define <vscale x 16 x i8> @bsl1n_i8(<vscale x 16 x i8> %a,
                                    <vscale x 16 x i8> %b,
                                    <vscale x 16 x i8> %c) {
; CHECK-LABEL: bsl1n_i8
; CHECK: bsl1n z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.bsl1n.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @bsl1n_i16(<vscale x 8 x i16> %a,
                                     <vscale x 8 x i16> %b,
                                     <vscale x 8 x i16> %c) {
; CHECK-LABEL: bsl1n_i16
; CHECK: bsl1n z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.bsl1n.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @bsl1n_i32(<vscale x 4 x i32> %a,
                                     <vscale x 4 x i32> %b,
                                     <vscale x 4 x i32> %c) {
; CHECK-LABEL: bsl1n_i32
; CHECK: bsl1n z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.bsl1n.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @bsl1n_i64(<vscale x 2 x i64> %a,
                                     <vscale x 2 x i64> %b,
                                     <vscale x 2 x i64> %c) {
; CHECK-LABEL: bsl1n_i64
; CHECK: bsl1n z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.bsl1n.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %res
}

;
; BSL2N (vector, bitwise, unpredicated)
;
define <vscale x 16 x i8> @bsl2n_i8(<vscale x 16 x i8> %a,
                                    <vscale x 16 x i8> %b,
                                    <vscale x 16 x i8> %c) {
; CHECK-LABEL: bsl2n_i8
; CHECK: bsl2n z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.bsl2n.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @bsl2n_i16(<vscale x 8 x i16> %a,
                                     <vscale x 8 x i16> %b,
                                     <vscale x 8 x i16> %c) {
; CHECK-LABEL: bsl2n_i16
; CHECK: bsl2n z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.bsl2n.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @bsl2n_i32(<vscale x 4 x i32> %a,
                                     <vscale x 4 x i32> %b,
                                     <vscale x 4 x i32> %c) {
; CHECK-LABEL: bsl2n_i32
; CHECK: bsl2n z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.bsl2n.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @bsl2n_i64(<vscale x 2 x i64> %a,
                                     <vscale x 2 x i64> %b,
                                     <vscale x 2 x i64> %c) {
; CHECK-LABEL: bsl2n_i64
; CHECK: bsl2n z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.bsl2n.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %res
}

;
; NBSL (vector, bitwise, unpredicated)
;
define <vscale x 16 x i8> @nbsl_i8(<vscale x 16 x i8> %a,
                                   <vscale x 16 x i8> %b,
                                   <vscale x 16 x i8> %c) {
; CHECK-LABEL: nbsl_i8
; CHECK: nbsl z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.nbsl.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @nbsl_i16(<vscale x 8 x i16> %a,
                                    <vscale x 8 x i16> %b,
                                    <vscale x 8 x i16> %c) {
; CHECK-LABEL: nbsl_i16
; CHECK: nbsl z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.nbsl.nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @nbsl_i32(<vscale x 4 x i32> %a,
                                    <vscale x 4 x i32> %b,
                                    <vscale x 4 x i32> %c) {
; CHECK-LABEL: nbsl_i32
; CHECK: nbsl z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.nbsl.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @nbsl_i64(<vscale x 2 x i64> %a,
                                    <vscale x 2 x i64> %b,
                                    <vscale x 2 x i64> %c) {
; CHECK-LABEL: nbsl_i64
; CHECK: nbsl z0.d, z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.nbsl.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %res
}

;
; XAR (vector, bitwise, unpredicated)
;

define <vscale x 16 x i8> @xar_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: xar_b:
; CHECK: xar z0.b, z0.b, z1.b, #1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.xar.nxv16i8(<vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b,
                                                               i32 1)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @xar_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: xar_h:
; CHECK: xar z0.h, z0.h, z1.h, #2
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.xar.nxv8i16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b,
                                                               i32 2)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @xar_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: xar_s:
; CHECK: xar z0.s, z0.s, z1.s, #3
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.xar.nxv4i32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b,
                                                               i32 3)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @xar_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: xar_d:
; CHECK: xar z0.d, z0.d, z1.d, #4
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.xar.nxv2i64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b,
                                                               i32 4)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.eor3.nxv16i8(<vscale x 16 x i8>,<vscale x 16 x i8>,<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.eor3.nxv8i16(<vscale x 8 x i16>,<vscale x 8 x i16>,<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.eor3.nxv4i32(<vscale x 4 x i32>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.eor3.nxv2i64(<vscale x 2 x i64>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.bcax.nxv16i8(<vscale x 16 x i8>,<vscale x 16 x i8>,<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.bcax.nxv8i16(<vscale x 8 x i16>,<vscale x 8 x i16>,<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.bcax.nxv4i32(<vscale x 4 x i32>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.bcax.nxv2i64(<vscale x 2 x i64>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.bsl.nxv16i8(<vscale x 16 x i8>,<vscale x 16 x i8>,<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.bsl.nxv8i16(<vscale x 8 x i16>,<vscale x 8 x i16>,<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.bsl.nxv4i32(<vscale x 4 x i32>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.bsl.nxv2i64(<vscale x 2 x i64>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.bsl1n.nxv16i8(<vscale x 16 x i8>,<vscale x 16 x i8>,<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.bsl1n.nxv8i16(<vscale x 8 x i16>,<vscale x 8 x i16>,<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.bsl1n.nxv4i32(<vscale x 4 x i32>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.bsl1n.nxv2i64(<vscale x 2 x i64>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.bsl2n.nxv16i8(<vscale x 16 x i8>,<vscale x 16 x i8>,<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.bsl2n.nxv8i16(<vscale x 8 x i16>,<vscale x 8 x i16>,<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.bsl2n.nxv4i32(<vscale x 4 x i32>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.bsl2n.nxv2i64(<vscale x 2 x i64>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.nbsl.nxv16i8(<vscale x 16 x i8>,<vscale x 16 x i8>,<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.nbsl.nxv8i16(<vscale x 8 x i16>,<vscale x 8 x i16>,<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.nbsl.nxv4i32(<vscale x 4 x i32>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.nbsl.nxv2i64(<vscale x 2 x i64>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.xar.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.xar.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.xar.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.xar.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, i32)
