; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; CNOT
;

define <vscale x 16 x i8> @cnot_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cnot_i8:
; CHECK: cnot z0.b, p0/m, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.cnot.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i1> %pg,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @cnot_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cnot_i16:
; CHECK: cnot z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.cnot.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @cnot_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cnot_i32:
; CHECK: cnot z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.cnot.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @cnot_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cnot_i64:
; CHECK: cnot z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.cnot.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; NOT
;

define <vscale x 16 x i8> @not_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg, <vscale x 16 x i8> %b) {
; CHECK-LABEL: not_i8:
; CHECK: not z0.b, p0/m, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.not.nxv16i8(<vscale x 16 x i8> %a,
                                                               <vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @not_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x i16> %b) {
; CHECK-LABEL: not_i16:
; CHECK: not z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.not.nxv8i16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @not_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: not_i32:
; CHECK: not z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.not.nxv4i32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @not_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: not_i64:
; CHECK: not z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.not.nxv2i64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.cnot.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.cnot.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.cnot.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.cnot.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.not.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.not.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.not.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.not.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)
