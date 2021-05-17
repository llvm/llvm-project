; Partial expansion cases (still VP with parameter expansions).
; RUN: opt --expandvp --expandvp-override-evl-transform=Legal --expandvp-override-mask-transform=Legal -S < %s | FileCheck %s --check-prefix=LEGAL_LEGAL
; RUN: opt --expandvp --expandvp-override-evl-transform=Discard --expandvp-override-mask-transform=Legal -S < %s | FileCheck %s --check-prefix=DISCARD_LEGAL
; RUN: opt --expandvp --expandvp-override-evl-transform=Convert --expandvp-override-mask-transform=Legal -S < %s | FileCheck %s --check-prefix=CONVERT_LEGAL
; Full expansion cases (all expanded to non-VP).
; RUN: opt --expandvp --expandvp-override-evl-transform=Discard --expandvp-override-mask-transform=Convert -S < %s | FileCheck %s --check-prefix=ALL-CONVERT
; RUN: opt --expandvp -S < %s | FileCheck %s --check-prefix=ALL-CONVERT
; RUN: opt --expandvp --expandvp-override-evl-transform=Legal --expandvp-override-mask-transform=Convert -S < %s | FileCheck %s --check-prefix=ALL-CONVERT
; RUN: opt --expandvp --expandvp-override-evl-transform=Convert --expandvp-override-mask-transform=Convert -S < %s | FileCheck %s --check-prefix=ALL-CONVERT


; Fixed-width vectors
; Integer arith
declare <8 x i32> @llvm.vp.add.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.sub.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.mul.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.srem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.urem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
; Bit arith
declare <8 x i32> @llvm.vp.and.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.xor.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.or.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.shl.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)

; Fixed vector test function.
define void @test_vp_int_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x i32> %i2, <8 x i32> %f3, <8 x i1> %m, i32 %n) {
  %r0 = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r1 = call <8 x i32> @llvm.vp.sub.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r2 = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r3 = call <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r4 = call <8 x i32> @llvm.vp.srem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r5 = call <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r6 = call <8 x i32> @llvm.vp.urem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r7 = call <8 x i32> @llvm.vp.and.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r8 = call <8 x i32> @llvm.vp.or.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r9 = call <8 x i32> @llvm.vp.xor.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rA = call <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rB = call <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rC = call <8 x i32> @llvm.vp.shl.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  ret void
}

; Scalable-width vectors
; Integer arith
declare <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.sub.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.mul.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.sdiv.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.srem.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.udiv.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.urem.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
; Bit arith
declare <vscale x 4 x i32> @llvm.vp.and.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.xor.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.or.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.ashr.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.lshr.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.shl.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)

; Scalable vector test function.
define void @test_vp_int_vscale(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i32> %i2, <vscale x 4 x i32> %f3, <vscale x 4 x i1> %m, i32 %n) {
  %r0 = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r1 = call <vscale x 4 x i32> @llvm.vp.sub.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r2 = call <vscale x 4 x i32> @llvm.vp.mul.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r3 = call <vscale x 4 x i32> @llvm.vp.sdiv.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r4 = call <vscale x 4 x i32> @llvm.vp.srem.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r5 = call <vscale x 4 x i32> @llvm.vp.udiv.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r6 = call <vscale x 4 x i32> @llvm.vp.urem.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r7 = call <vscale x 4 x i32> @llvm.vp.and.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r8 = call <vscale x 4 x i32> @llvm.vp.or.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r9 = call <vscale x 4 x i32> @llvm.vp.xor.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %rA = call <vscale x 4 x i32> @llvm.vp.ashr.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %rB = call <vscale x 4 x i32> @llvm.vp.lshr.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %rC = call <vscale x 4 x i32> @llvm.vp.shl.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  ret void
}
; All VP intrinsics have to be lowered into non-VP ops
; Convert %evl into %mask for non-speculatable VP intrinsics and emit the
; instruction+select idiom with a non-VP SIMD instruction.
;
; ALL-CONVERT-NOT: {{call.* @llvm.vp.add}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.sub}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.mul}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.sdiv}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.srem}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.udiv}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.urem}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.and}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.or}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.xor}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.ashr}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.lshr}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.shl}}
;
; ALL-CONVERT: define void @test_vp_int_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x i32> %i2, <8 x i32> %f3, <8 x i1> %m, i32 %n) {
; ALL-CONVERT-NEXT:  %{{.*}} = add <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.*}} = sub <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.*}} = mul <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  [[NINS:%.+]] = insertelement <8 x i32> poison, i32 %n, i32 0
; ALL-CONVERT-NEXT:  [[NSPLAT:%.+]] = shufflevector <8 x i32> [[NINS]], <8 x i32> poison, <8 x i32> zeroinitializer
; ALL-CONVERT-NEXT:  [[EVLM:%.+]] = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[NSPLAT]]
; ALL-CONVERT-NEXT:  [[NEWM:%.+]] = and <8 x i1> [[EVLM]], %m
; ALL-CONVERT-NEXT:  [[SELONE:%.+]] = select <8 x i1> [[NEWM]], <8 x i32> %i1, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
; ALL-CONVERT-NEXT:  %{{.+}} = sdiv <8 x i32> %i0, [[SELONE]]
; ALL-CONVERT-NOT:   %{{.+}} = srem <8 x i32> %i0, %i1
; ALL-CONVERT:       %{{.+}} = srem <8 x i32> %i0, %{{.+}}
; ALL-CONVERT-NOT:   %{{.+}} = udiv <8 x i32> %i0, %i1
; ALL-CONVERT:       %{{.+}} = udiv <8 x i32> %i0, %{{.+}}
; ALL-CONVERT-NOT:   %{{.+}} = urem <8 x i32> %i0, %i1
; ALL-CONVERT:       %{{.+}} = urem <8 x i32> %i0, %{{.+}}
; ALL-CONVERT-NEXT:  %{{.+}} = and <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.+}} = or <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.+}} = xor <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.+}} = ashr <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.+}} = lshr <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.+}} = shl <8 x i32> %i0, %i1
; ALL-CONVERT:       ret void




; All legal - don't transform anything.

; LEGAL_LEGAL: define void @test_vp_int_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x i32> %i2, <8 x i32> %f3, <8 x i1> %m, i32 %n) {
; LEGAL_LEGAL-NEXT:   %r0 = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r1 = call <8 x i32> @llvm.vp.sub.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r2 = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r3 = call <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r4 = call <8 x i32> @llvm.vp.srem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r5 = call <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r6 = call <8 x i32> @llvm.vp.urem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r7 = call <8 x i32> @llvm.vp.and.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r8 = call <8 x i32> @llvm.vp.or.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r9 = call <8 x i32> @llvm.vp.xor.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %rA = call <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %rB = call <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %rC = call <8 x i32> @llvm.vp.shl.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   ret void

; LEGAL_LEGAL:define void @test_vp_int_vscale(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i32> %i2, <vscale x 4 x i32> %f3, <vscale x 4 x i1> %m, i32 %n) {
; LEGAL_LEGAL-NEXT:  %r0 = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r1 = call <vscale x 4 x i32> @llvm.vp.sub.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r2 = call <vscale x 4 x i32> @llvm.vp.mul.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r3 = call <vscale x 4 x i32> @llvm.vp.sdiv.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r4 = call <vscale x 4 x i32> @llvm.vp.srem.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r5 = call <vscale x 4 x i32> @llvm.vp.udiv.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r6 = call <vscale x 4 x i32> @llvm.vp.urem.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r7 = call <vscale x 4 x i32> @llvm.vp.and.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r8 = call <vscale x 4 x i32> @llvm.vp.or.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r9 = call <vscale x 4 x i32> @llvm.vp.xor.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %rA = call <vscale x 4 x i32> @llvm.vp.ashr.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %rB = call <vscale x 4 x i32> @llvm.vp.lshr.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %rC = call <vscale x 4 x i32> @llvm.vp.shl.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  ret void


; Drop %evl where possible else fold %evl into %mask (%evl Discard, %mask Legal)
;
; There is no caching yet in the ExpandVectorPredication pass and the %evl
; expansion code is emitted for every non-speculatable intrinsic again. Hence,
; only check that..
; (1) The %evl folding code and %mask are correct for the first
;     non-speculatable VP intrinsic.
; (2) All other non-speculatable VP intrinsics have a modified mask argument.
; (3) All speculatable VP intrinsics keep their %mask and %evl.
; (4) All VP intrinsics have an ineffective %evl parameter.

; DISCARD_LEGAL: define void @test_vp_int_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x i32> %i2, <8 x i32> %f3, <8 x i1> %m, i32 %n) {
; DISCARD_LEGAL-NEXT:   %r0 = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %r1 = call <8 x i32> @llvm.vp.sub.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %r2 = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   [[NSPLATINS:%.+]] = insertelement <8 x i32> poison, i32 %n, i32 0
; DISCARD_LEGAL-NEXT:   [[NSPLAT:%.+]] = shufflevector <8 x i32> [[NSPLATINS]], <8 x i32> poison, <8 x i32> zeroinitializer
; DISCARD_LEGAL-NEXT:   [[EVLMASK:%.+]] = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[NSPLAT]]
; DISCARD_LEGAL-NEXT:   [[NEWMASK:%.+]] = and <8 x i1> [[EVLMASK]], %m
; DISCARD_LEGAL-NEXT:   %r3 = call <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> [[NEWMASK]], i32 8)
; DISCARD_LEGAL-NOT:    %r4 = call <8 x i32> @llvm.vp.srem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NOT:    %r5 = call <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NOT:    %r6 = call <8 x i32> @llvm.vp.urem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL:        %r7 = call <8 x i32> @llvm.vp.and.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %r8 = call <8 x i32> @llvm.vp.or.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %r9 = call <8 x i32> @llvm.vp.xor.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %rA = call <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %rB = call <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %rC = call <8 x i32> @llvm.vp.shl.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   ret void

; TODO compute vscale only once and use caching.
; In the meantime, we only check for the correct vscale code for the first VP
; intrinsic and skip over it for all others.

; DISCARD_LEGAL: define void @test_vp_int_vscale(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i32> %i2, <vscale x 4 x i32> %f3, <vscale x 4 x i1> %m, i32 %n) {
; DISCARD_LEGAL-NEXT: %vscale = call i32 @llvm.vscale.i32()
; DISCARD_LEGAL-NEXT: %scalable_size = mul nuw i32 %vscale, 4
; DISCARD_LEGAL-NEXT: %r0 = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %scalable_size)
; DISCARD_LEGAL:      %r1 = call <vscale x 4 x i32> @llvm.vp.sub.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %scalable_size{{.*}})
; DISCARD_LEGAL:      %r2 = call <vscale x 4 x i32> @llvm.vp.mul.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %scalable_size{{.*}})
; DISCARD_LEGAL:      [[EVLM:%.+]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 0, i32 %n)
; DISCARD_LEGAL:      [[NEWM:%.+]] = and <vscale x 4 x i1> [[EVLM]], %m
; DISCARD_LEGAL:      %r3 = call <vscale x 4 x i32> @llvm.vp.sdiv.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> [[NEWM]], i32 %scalable_size{{.*}})
; DISCARD_LEGAL-NOT:  %{{.+}} = call <vscale x 4 x i32> @llvm.vp.{{.*}}, i32 %n)
; DISCARD_LEGAL:      ret void


; Convert %evl into %mask everywhere (%evl Convert, %mask Legal)
;
; For the same reasons as in the (%evl Discard, %mask Legal) case only check that..
; (1) The %evl folding code and %mask are correct for the first VP intrinsic.
; (2) All other VP intrinsics have a modified mask argument.
; (3) All VP intrinsics have an ineffective %evl parameter.
;
; CONVERT_LEGAL: define void @test_vp_int_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x i32> %i2, <8 x i32> %f3, <8 x i1> %m, i32 %n) {
; CONVERT_LEGAL-NEXT:  [[NINS:%.+]] = insertelement <8 x i32> poison, i32 %n, i32 0
; CONVERT_LEGAL-NEXT:  [[NSPLAT:%.+]] = shufflevector <8 x i32> [[NINS]], <8 x i32> poison, <8 x i32> zeroinitializer
; CONVERT_LEGAL-NEXT:  [[EVLM:%.+]] = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[NSPLAT]]
; CONVERT_LEGAL-NEXT:  [[NEWM:%.+]] = and <8 x i1> [[EVLM]], %m
; CONVERT_LEGAL-NEXT:  %{{.+}} = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> [[NEWM]], i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.sub.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.srem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.urem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.and.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.or.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.xor.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.shl.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL:       ret void

; Similar to %evl discard, %mask legal but make sure the first VP intrinsic has a legal expansion
; CONVERT_LEGAL: define void @test_vp_int_vscale(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i32> %i2, <vscale x 4 x i32> %f3, <vscale x 4 x i1> %m, i32 %n) {
; CONVERT_LEGAL-NEXT:   [[EVLM:%.+]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 0, i32 %n)
; CONVERT_LEGAL-NEXT:   [[NEWM:%.+]] = and <vscale x 4 x i1> [[EVLM]], %m
; CONVERT_LEGAL-NEXT:   %vscale = call i32 @llvm.vscale.i32()
; CONVERT_LEGAL-NEXT:   %scalable_size = mul nuw i32 %vscale, 4
; CONVERT_LEGAL-NEXT:   %r0 = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> [[NEWM]], i32 %scalable_size)
; CONVERT_LEGAL-NOT:    %{{.*}} = call <vscale x 4 x i32> @llvm.vp.{{.*}}, i32 %n)
; CONVERT_LEGAL:        ret void

