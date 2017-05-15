; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a57 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKA57
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a72 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKA72
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m1  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKM1

declare <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %d, <16 x i8> %k)
declare <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %d)
declare <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %d, <16 x i8> %k)
declare <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %d)

define void @aesea(<16 x i8>* %a0, <16 x i8>* %b0, <16 x i8>* %c0, <16 x i8> %d, <16 x i8> %e) {
  %d0 = load <16 x i8>, <16 x i8>* %a0
  %a1 = getelementptr inbounds <16 x i8>, <16 x i8>* %a0, i64 1
  %d1 = load <16 x i8>, <16 x i8>* %a1
  %a2 = getelementptr inbounds <16 x i8>, <16 x i8>* %a0, i64 2
  %d2 = load <16 x i8>, <16 x i8>* %a2
  %a3 = getelementptr inbounds <16 x i8>, <16 x i8>* %a0, i64 3
  %d3 = load <16 x i8>, <16 x i8>* %a3
  %k0 = load <16 x i8>, <16 x i8>* %b0
  %e00 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %d0, <16 x i8> %k0)
  %f00 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e00)
  %e01 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %d1, <16 x i8> %k0)
  %f01 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e01)
  %e02 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %d2, <16 x i8> %k0)
  %f02 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e02)
  %e03 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %d3, <16 x i8> %k0)
  %f03 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e03)
  %b1 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 1
  %k1 = load <16 x i8>, <16 x i8>* %b1
  %e10 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f00, <16 x i8> %k1)
  %f10 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e00)
  %e11 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f01, <16 x i8> %k1)
  %f11 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e01)
  %e12 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f02, <16 x i8> %k1)
  %f12 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e02)
  %e13 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f03, <16 x i8> %k1)
  %f13 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e03)
  %b2 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 2
  %k2 = load <16 x i8>, <16 x i8>* %b2
  %e20 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f10, <16 x i8> %k2)
  %f20 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e10)
  %e21 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f11, <16 x i8> %k2)
  %f21 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e11)
  %e22 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f12, <16 x i8> %k2)
  %f22 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e12)
  %e23 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f13, <16 x i8> %k2)
  %f23 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e13)
  %b3 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 3
  %k3 = load <16 x i8>, <16 x i8>* %b3
  %e30 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f20, <16 x i8> %k3)
  %f30 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e20)
  %e31 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f21, <16 x i8> %k3)
  %f31 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e21)
  %e32 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f22, <16 x i8> %k3)
  %f32 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e22)
  %e33 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f23, <16 x i8> %k3)
  %f33 = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> %e23)
  %g0 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f30, <16 x i8> %d)
  %h0 = xor <16 x i8> %g0, %e
  %g1 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f31, <16 x i8> %d)
  %h1 = xor <16 x i8> %g1, %e
  %g2 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f32, <16 x i8> %d)
  %h2 = xor <16 x i8> %g2, %e
  %g3 = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %f33, <16 x i8> %d)
  %h3 = xor <16 x i8> %g3, %e
  store <16 x i8> %h0, <16 x i8>* %c0
  %c1 = getelementptr inbounds <16 x i8>, <16 x i8>* %c0, i64 1
  store <16 x i8> %h1, <16 x i8>* %c1
  %c2 = getelementptr inbounds <16 x i8>, <16 x i8>* %c0, i64 2
  store <16 x i8> %h2, <16 x i8>* %c2
  %c3 = getelementptr inbounds <16 x i8>, <16 x i8>* %c0, i64 3
  store <16 x i8> %h3, <16 x i8>* %c3
  ret void

; CHECK-LABEL: aesea:
; CHECKA57: aese [[VA:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57: aese [[VB:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57: aese [[VC:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesmc {{v[0-7].16b}}, [[VC]]
; CHECKA57: aesmc {{v[0-7].16b}}, [[VA]]
; CHECKA57: aese [[VD:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesmc {{v[0-7].16b}}, [[VD]]
; CHECKA57: aesmc {{v[0-7].16b}}, [[VB]]
; CHECKA57: aese [[VE:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesmc {{v[0-7].16b}}, [[VE]]
; CHECKA57: aese [[VF:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesmc {{v[0-7].16b}}, [[VF]]
; CHECKA57: aese [[VG:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesmc {{v[0-7].16b}}, [[VG]]
; CHECKA57: aese [[VH:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesmc {{v[0-7].16b}}, [[VH]]
; CHECKA72: aese [[VA:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesmc {{v[0-7].16b}}, [[VA]]
; CHECKA72: aese [[VB:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesmc {{v[0-7].16b}}, [[VB]]
; CHECKA72: aese [[VC:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesmc {{v[0-7].16b}}, [[VC]]
; CHECKA72: aese [[VD:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesmc {{v[0-7].16b}}, [[VD]]
; CHECKA72: aese [[VE:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesmc {{v[0-7].16b}}, [[VE]]
; CHECKA72: aese [[VF:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesmc {{v[0-7].16b}}, [[VF]]
; CHECKA72: aese [[VG:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesmc {{v[0-7].16b}}, [[VG]]
; CHECKA72: aese [[VH:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesmc {{v[0-7].16b}}, [[VH]]
; CHECKM1: aese [[VA:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1: aesmc {{v[0-7].16b}}, [[VA]]
; CHECKM1: aese [[VB:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesmc {{v[0-7].16b}}, [[VB]]
; CHECKM1: aese {{v[0-7].16b}}, {{v[0-7].16b}}
; CHECKM1: aese [[VC:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesmc {{v[0-7].16b}}, [[VC]]
; CHECKM1: aese [[VD:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1: aesmc {{v[0-7].16b}}, [[VD]]
; CHECKM1: aese [[VE:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesmc {{v[0-7].16b}}, [[VE]]
; CHECKM1: aese [[VF:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesmc {{v[0-7].16b}}, [[VF]]
; CHECKM1: aese [[VG:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesmc {{v[0-7].16b}}, [[VG]]
; CHECKM1: aese [[VH:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesmc {{v[0-7].16b}}, [[VH]]
}

define void @aesda(<16 x i8>* %a0, <16 x i8>* %b0, <16 x i8>* %c0, <16 x i8> %d, <16 x i8> %e) {
  %d0 = load <16 x i8>, <16 x i8>* %a0
  %a1 = getelementptr inbounds <16 x i8>, <16 x i8>* %a0, i64 1
  %d1 = load <16 x i8>, <16 x i8>* %a1
  %a2 = getelementptr inbounds <16 x i8>, <16 x i8>* %a0, i64 2
  %d2 = load <16 x i8>, <16 x i8>* %a2
  %a3 = getelementptr inbounds <16 x i8>, <16 x i8>* %a0, i64 3
  %d3 = load <16 x i8>, <16 x i8>* %a3
  %k0 = load <16 x i8>, <16 x i8>* %b0
  %e00 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %d0, <16 x i8> %k0)
  %f00 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e00)
  %e01 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %d1, <16 x i8> %k0)
  %f01 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e01)
  %e02 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %d2, <16 x i8> %k0)
  %f02 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e02)
  %e03 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %d3, <16 x i8> %k0)
  %f03 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e03)
  %b1 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 1
  %k1 = load <16 x i8>, <16 x i8>* %b1
  %e10 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f00, <16 x i8> %k1)
  %f10 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e00)
  %e11 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f01, <16 x i8> %k1)
  %f11 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e01)
  %e12 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f02, <16 x i8> %k1)
  %f12 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e02)
  %e13 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f03, <16 x i8> %k1)
  %f13 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e03)
  %b2 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 2
  %k2 = load <16 x i8>, <16 x i8>* %b2
  %e20 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f10, <16 x i8> %k2)
  %f20 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e10)
  %e21 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f11, <16 x i8> %k2)
  %f21 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e11)
  %e22 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f12, <16 x i8> %k2)
  %f22 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e12)
  %e23 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f13, <16 x i8> %k2)
  %f23 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e13)
  %b3 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 3
  %k3 = load <16 x i8>, <16 x i8>* %b3
  %e30 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f20, <16 x i8> %k3)
  %f30 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e20)
  %e31 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f21, <16 x i8> %k3)
  %f31 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e21)
  %e32 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f22, <16 x i8> %k3)
  %f32 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e22)
  %e33 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f23, <16 x i8> %k3)
  %f33 = call <16 x i8> @llvm.aarch64.crypto.aesimc(<16 x i8> %e23)
  %g0 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f30, <16 x i8> %d)
  %h0 = xor <16 x i8> %g0, %e
  %g1 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f31, <16 x i8> %d)
  %h1 = xor <16 x i8> %g1, %e
  %g2 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f32, <16 x i8> %d)
  %h2 = xor <16 x i8> %g2, %e
  %g3 = call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %f33, <16 x i8> %d)
  %h3 = xor <16 x i8> %g3, %e
  store <16 x i8> %h0, <16 x i8>* %c0
  %c1 = getelementptr inbounds <16 x i8>, <16 x i8>* %c0, i64 1
  store <16 x i8> %h1, <16 x i8>* %c1
  %c2 = getelementptr inbounds <16 x i8>, <16 x i8>* %c0, i64 2
  store <16 x i8> %h2, <16 x i8>* %c2
  %c3 = getelementptr inbounds <16 x i8>, <16 x i8>* %c0, i64 3
  store <16 x i8> %h3, <16 x i8>* %c3
  ret void

; CHECK-LABEL: aesda:
; CHECKA57: aesd [[VA:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57: aesd [[VB:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57: aesd [[VC:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesimc {{v[0-7].16b}}, [[VC]]
; CHECKA57: aesimc {{v[0-7].16b}}, [[VA]]
; CHECKA57: aesd [[VD:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesimc {{v[0-7].16b}}, [[VD]]
; CHECKA57: aesimc {{v[0-7].16b}}, [[VB]]
; CHECKA57: aesd [[VE:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesimc {{v[0-7].16b}}, [[VE]]
; CHECKA57: aesd [[VF:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesimc {{v[0-7].16b}}, [[VF]]
; CHECKA57: aesd [[VG:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesimc {{v[0-7].16b}}, [[VG]]
; CHECKA57: aesd [[VH:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA57-NEXT: aesimc {{v[0-7].16b}}, [[VH]]
; CHECKA72: aesd [[VA:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesimc {{v[0-7].16b}}, [[VA]]
; CHECKA72: aesd [[VB:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesimc {{v[0-7].16b}}, [[VB]]
; CHECKA72: aesd [[VC:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesimc {{v[0-7].16b}}, [[VC]]
; CHECKA72: aesd [[VD:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesimc {{v[0-7].16b}}, [[VD]]
; CHECKA72: aesd [[VE:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesimc {{v[0-7].16b}}, [[VE]]
; CHECKA72: aesd [[VF:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesimc {{v[0-7].16b}}, [[VF]]
; CHECKA72: aesd [[VG:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesimc {{v[0-7].16b}}, [[VG]]
; CHECKA72: aesd [[VH:v[0-7].16b]], {{v[0-7].16b}}
; CHECKA72-NEXT: aesimc {{v[0-7].16b}}, [[VH]]
; CHECKM1: aesd [[VA:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1: aesimc {{v[0-7].16b}}, [[VA]]
; CHECKM1: aesd [[VB:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesimc {{v[0-7].16b}}, [[VB]]
; CHECKM1: aesd {{v[0-7].16b}}, {{v[0-7].16b}}
; CHECKM1: aesd [[VC:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesimc {{v[0-7].16b}}, [[VC]]
; CHECKM1: aesd [[VD:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1: aesimc {{v[0-7].16b}}, [[VD]]
; CHECKM1: aesd [[VE:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesimc {{v[0-7].16b}}, [[VE]]
; CHECKM1: aesd [[VF:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesimc {{v[0-7].16b}}, [[VF]]
; CHECKM1: aesd [[VG:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesimc {{v[0-7].16b}}, [[VG]]
; CHECKM1: aesd [[VH:v[0-7].16b]], {{v[0-7].16b}}
; CHECKM1-NEXT: aesimc {{v[0-7].16b}}, [[VH]]
}
