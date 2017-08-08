; RUN: llc %s -o - -mtriple=armv8 -mattr=+crypto,+fuse-aes -enable-misched -disable-post-ra | FileCheck %s

declare <16 x i8> @llvm.arm.neon.aese(<16 x i8> %d, <16 x i8> %k)
declare <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %d)
declare <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %d, <16 x i8> %k)
declare <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %d)

define void @aesea(<16 x i8>* %a0, <16 x i8>* %b0, <16 x i8>* %c0, <16 x i8> %d, <16 x i8> %e) {
  %d0 = load <16 x i8>, <16 x i8>* %a0
  %a1 = getelementptr inbounds <16 x i8>, <16 x i8>* %a0, i64 1
  %d1 = load <16 x i8>, <16 x i8>* %a1
  %a2 = getelementptr inbounds <16 x i8>, <16 x i8>* %a0, i64 2
  %d2 = load <16 x i8>, <16 x i8>* %a2
  %a3 = getelementptr inbounds <16 x i8>, <16 x i8>* %a0, i64 3
  %d3 = load <16 x i8>, <16 x i8>* %a3
  %k0 = load <16 x i8>, <16 x i8>* %b0
  %e00 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %d0, <16 x i8> %k0)
  %f00 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e00)
  %e01 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %d1, <16 x i8> %k0)
  %f01 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e01)
  %e02 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %d2, <16 x i8> %k0)
  %f02 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e02)
  %e03 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %d3, <16 x i8> %k0)
  %f03 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e03)
  %b1 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 1
  %k1 = load <16 x i8>, <16 x i8>* %b1
  %e10 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f00, <16 x i8> %k1)
  %f10 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e00)
  %e11 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f01, <16 x i8> %k1)
  %f11 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e01)
  %e12 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f02, <16 x i8> %k1)
  %f12 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e02)
  %e13 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f03, <16 x i8> %k1)
  %f13 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e03)
  %b2 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 2
  %k2 = load <16 x i8>, <16 x i8>* %b2
  %e20 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f10, <16 x i8> %k2)
  %f20 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e10)
  %e21 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f11, <16 x i8> %k2)
  %f21 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e11)
  %e22 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f12, <16 x i8> %k2)
  %f22 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e12)
  %e23 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f13, <16 x i8> %k2)
  %f23 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e13)
  %b3 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 3
  %k3 = load <16 x i8>, <16 x i8>* %b3
  %e30 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f20, <16 x i8> %k3)
  %f30 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e20)
  %e31 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f21, <16 x i8> %k3)
  %f31 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e21)
  %e32 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f22, <16 x i8> %k3)
  %f32 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e22)
  %e33 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f23, <16 x i8> %k3)
  %f33 = call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %e23)
  %g0 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f30, <16 x i8> %d)
  %h0 = xor <16 x i8> %g0, %e
  %g1 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f31, <16 x i8> %d)
  %h1 = xor <16 x i8> %g1, %e
  %g2 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f32, <16 x i8> %d)
  %h2 = xor <16 x i8> %g2, %e
  %g3 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %f33, <16 x i8> %d)
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
; CHECK: aese.8 [[QA:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesmc.8 {{q[0-9][0-9]?}}, [[QA]]
; CHECK: aese.8 [[QB:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesmc.8 {{q[0-9][0-9]?}}, [[QB]]
; CHECK: aese.8 {{q[0-9][0-9]?}}, {{q[0-9][0-9]?}}
; CHECK: aese.8 [[QC:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesmc.8 {{q[0-9][0-9]?}}, [[QC]]
; CHECK: aese.8 [[QD:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesmc.8 {{q[0-9][0-9]?}}, [[QD]]
; CHECK: aese.8 [[QE:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesmc.8 {{q[0-9][0-9]?}}, [[QE]]
; CHECK: aese.8 {{q[0-9][0-9]?}}, {{q[0-9][0-9]?}}
; CHECK: aese.8 [[QF:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesmc.8 {{q[0-9][0-9]?}}, [[QF]]
; CHECK: aese.8 [[QG:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesmc.8 {{q[0-9][0-9]?}}, [[QG]]
; CHECK: aese.8 {{q[0-9][0-9]?}}, {{q[0-9][0-9]?}}
; CHECK: aese.8 [[QH:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesmc.8 {{q[0-9][0-9]?}}, [[QH]]
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
  %e00 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %d0, <16 x i8> %k0)
  %f00 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e00)
  %e01 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %d1, <16 x i8> %k0)
  %f01 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e01)
  %e02 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %d2, <16 x i8> %k0)
  %f02 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e02)
  %e03 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %d3, <16 x i8> %k0)
  %f03 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e03)
  %b1 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 1
  %k1 = load <16 x i8>, <16 x i8>* %b1
  %e10 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f00, <16 x i8> %k1)
  %f10 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e00)
  %e11 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f01, <16 x i8> %k1)
  %f11 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e01)
  %e12 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f02, <16 x i8> %k1)
  %f12 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e02)
  %e13 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f03, <16 x i8> %k1)
  %f13 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e03)
  %b2 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 2
  %k2 = load <16 x i8>, <16 x i8>* %b2
  %e20 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f10, <16 x i8> %k2)
  %f20 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e10)
  %e21 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f11, <16 x i8> %k2)
  %f21 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e11)
  %e22 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f12, <16 x i8> %k2)
  %f22 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e12)
  %e23 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f13, <16 x i8> %k2)
  %f23 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e13)
  %b3 = getelementptr inbounds <16 x i8>, <16 x i8>* %b0, i64 3
  %k3 = load <16 x i8>, <16 x i8>* %b3
  %e30 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f20, <16 x i8> %k3)
  %f30 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e20)
  %e31 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f21, <16 x i8> %k3)
  %f31 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e21)
  %e32 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f22, <16 x i8> %k3)
  %f32 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e22)
  %e33 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f23, <16 x i8> %k3)
  %f33 = call <16 x i8> @llvm.arm.neon.aesimc(<16 x i8> %e23)
  %g0 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f30, <16 x i8> %d)
  %h0 = xor <16 x i8> %g0, %e
  %g1 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f31, <16 x i8> %d)
  %h1 = xor <16 x i8> %g1, %e
  %g2 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f32, <16 x i8> %d)
  %h2 = xor <16 x i8> %g2, %e
  %g3 = call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %f33, <16 x i8> %d)
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
; CHECK: aesd.8 [[QA:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesimc.8 {{q[0-9][0-9]?}}, [[QA]]
; CHECK: aesd.8 [[QB:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesimc.8 {{q[0-9][0-9]?}}, [[QB]]
; CHECK: aesd.8 {{q[0-9][0-9]?}}, {{q[0-9][0-9]?}}
; CHECK: aesd.8 [[QC:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesimc.8 {{q[0-9][0-9]?}}, [[QC]]
; CHECK: aesd.8 [[QD:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesimc.8 {{q[0-9][0-9]?}}, [[QD]]
; CHECK: aesd.8 [[QE:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesimc.8 {{q[0-9][0-9]?}}, [[QE]]
; CHECK: aesd.8 {{q[0-9][0-9]?}}, {{q[0-9][0-9]?}}
; CHECK: aesd.8 [[QF:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesimc.8 {{q[0-9][0-9]?}}, [[QF]]
; CHECK: aesd.8 [[QG:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesimc.8 {{q[0-9][0-9]?}}, [[QG]]
; CHECK: aesd.8 {{q[0-9][0-9]?}}, {{q[0-9][0-9]?}}
; CHECK: aesd.8 [[QH:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesimc.8 {{q[0-9][0-9]?}}, [[QH]]
}

define void @aes_load_store(<16 x i8> *%p1, <16 x i8> *%p2 , <16 x i8> *%p3) {
entry:
  %x1 = alloca <16 x i8>, align 16
  %x2 = alloca <16 x i8>, align 16
  %x3 = alloca <16 x i8>, align 16
  %x4 = alloca <16 x i8>, align 16
  %x5 = alloca <16 x i8>, align 16
  %in1 = load <16 x i8>, <16 x i8>* %p1, align 16
  store <16 x i8> %in1, <16 x i8>* %x1, align 16
  %aese1 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %in1, <16 x i8> %in1) #2
  store <16 x i8> %aese1, <16 x i8>* %x2, align 16
  %in2 = load <16 x i8>, <16 x i8>* %p2, align 16
  %aesmc1= call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %aese1) #2
  store <16 x i8> %aesmc1, <16 x i8>* %x3, align 16
  %aese2 = call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %in1, <16 x i8> %in2) #2
  store <16 x i8> %aese2, <16 x i8>* %x4, align 16
  %aesmc2= call <16 x i8> @llvm.arm.neon.aesmc(<16 x i8> %aese2) #2
  store <16 x i8> %aesmc2, <16 x i8>* %x5, align 16
  ret void

; CHECK-LABEL: aes_load_store:
; CHECK: aese.8 [[QA:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesmc.8 {{q[0-9][0-9]?}}, [[QA]]
; CHECK: aese.8 [[QB:q[0-9][0-9]?]], {{q[0-9][0-9]?}}
; CHECK-NEXT: aesmc.8 {{q[0-9][0-9]?}}, [[QB]]
}
