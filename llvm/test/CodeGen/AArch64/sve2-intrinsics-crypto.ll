; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2-aes,+sve2-sha3,+sve2-sm4 -asm-verbose=0 < %s | FileCheck %s

;
; AESD
;

define <vscale x 16 x i8> @aesd_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: aesd_i8:
; CHECK: aesd z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.aesd(<vscale x 16 x i8> %a,
                                                        <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

;
; AESIMC
;

define <vscale x 16 x i8> @aesimc_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: aesimc_i8:
; CHECK: aesimc z0.b, z0.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.aesimc(<vscale x 16 x i8> %a)
  ret <vscale x 16 x i8> %out
}

;
; AESE
;

define <vscale x 16 x i8> @aese_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: aese_i8:
; CHECK: aese z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.aese(<vscale x 16 x i8> %a,
                                                        <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

;
; AESMC
;

define <vscale x 16 x i8> @aesmc_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: aesmc_i8:
; CHECK: aesmc z0.b, z0.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.aesmc(<vscale x 16 x i8> %a)
  ret <vscale x 16 x i8> %out
}

;
; RAX1
;

define <vscale x 2 x i64> @rax1_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: rax1_i64:
; CHECK: rax1 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.rax1(<vscale x 2 x i64> %a,
                                                        <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SM4E
;

define <vscale x 4 x i32> @sm4e_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sm4e_i32:
; CHECK: sm4e z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sm4e(<vscale x 4 x i32> %a,
                                                        <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

;
; SM4EKEY
;

define <vscale x 4 x i32> @sm4ekey_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sm4ekey_i32:
; CHECK: sm4ekey z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sm4ekey(<vscale x 4 x i32> %a,
                                                           <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}


declare <vscale x 16 x i8> @llvm.aarch64.sve.aesd(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.aesimc(<vscale x 16 x i8>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.aese(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.aesmc(<vscale x 16 x i8>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.rax1(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sm4e(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sm4ekey(<vscale x 4 x i32>, <vscale x 4 x i32>)
