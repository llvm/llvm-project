; RUN: llc < %s -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-min=256 | FileCheck %s
; RUN: llc < %s -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-min=256 -stop-after=finalize-isel | FileCheck %s --check-prefix=MIR

declare <8 x i16> @llvm.vp.merge.nxv2i16(<8 x i1>, <8 x i16>, <8 x i16>, i32)
declare <8 x i32> @llvm.vp.merge.nxv2i32(<8 x i1>, <8 x i32>, <8 x i32>, i32)
declare <8 x float> @llvm.vp.merge.nxv2f32(<8 x i1>, <8 x float>, <8 x float>, i32)
declare <8 x double> @llvm.vp.merge.nxv2f64(<8 x i1>, <8 x double>, <8 x double>, i32)

; Test binary operator with vp.merge and vp.smax.
declare <8 x i32> @llvm.vp.add.nxv2i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
define <8 x i32> @vpmerge_vpadd(<8 x i32> %passthru, <8 x i32> %x, <8 x i32> %y, <8 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpadd:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vadd.vv v8, v9, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpadd
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v9, $v10, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v10
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY4:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   [[PseudoVADD_VV_M1_MASK:%[0-9]+]]:vrnov0 = PseudoVADD_VV_M1_MASK [[COPY4]], [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY [[PseudoVADD_VV_M1_MASK]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x i32> @llvm.vp.add.nxv2i32(<8 x i32> %x, <8 x i32> %y, <8 x i1> %mask, i32 %vl)
  %b = call <8 x i32> @llvm.vp.merge.nxv2i32(<8 x i1> %m, <8 x i32> %a, <8 x i32> %passthru, i32 %vl)
  ret <8 x i32> %b
}

; Test glued node of merge should not be deleted.
declare <8 x i1> @llvm.vp.icmp.nxv2i32(<8 x i32>, <8 x i32>, metadata, <8 x i1>, i32)
define <8 x i32> @vpmerge_vpadd2(<8 x i32> %passthru, <8 x i32> %x, <8 x i32> %y, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpadd2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vmseq.vv v0, v9, v10
; CHECK-NEXT:    vsetvli zero, zero, e32, m1, tu, mu
; CHECK-NEXT:    vadd.vv v8, v9, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpadd2
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v9, $v10, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v10
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   [[PseudoVMSEQ_VV_M1_:%[0-9]+]]:vr = PseudoVMSEQ_VV_M1 [[COPY2]], [[COPY1]], [[COPY]], 5 /* e32 */
  ; MIR-NEXT:   $v0 = COPY [[PseudoVMSEQ_VV_M1_]]
  ; MIR-NEXT:   [[PseudoVADD_VV_M1_MASK:%[0-9]+]]:vrnov0 = PseudoVADD_VV_M1_MASK [[COPY3]], [[COPY2]], [[COPY1]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY [[PseudoVADD_VV_M1_MASK]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x i32> @llvm.vp.add.nxv2i32(<8 x i32> %x, <8 x i32> %y, <8 x i1> %mask, i32 %vl)
  %m = call <8 x i1> @llvm.vp.icmp.nxv2i32(<8 x i32> %x, <8 x i32> %y, metadata !"eq", <8 x i1> %mask, i32 %vl)
  %b = call <8 x i32> @llvm.vp.merge.nxv2i32(<8 x i1> %m, <8 x i32> %a, <8 x i32> %passthru, i32 %vl)
  ret <8 x i32> %b
}

; Test vp.merge have all-ones mask.
define <8 x i32> @vpmerge_vpadd3(<8 x i32> %passthru, <8 x i32> %x, <8 x i32> %y, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpadd3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vadd.vv v8, v9, v10
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpadd3
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v9, $v10, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v10
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vr = COPY $v8
  ; MIR-NEXT:   [[PseudoVADD_VV_M1_TU:%[0-9]+]]:vr = PseudoVADD_VV_M1_TU [[COPY3]], [[COPY2]], [[COPY1]], [[COPY]], 5 /* e32 */
  ; MIR-NEXT:   $v8 = COPY [[PseudoVADD_VV_M1_TU]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x i32> @llvm.vp.add.nxv2i32(<8 x i32> %x, <8 x i32> %y, <8 x i1> %mask, i32 %vl)
  %b = call <8 x i32> @llvm.vp.merge.nxv2i32(<8 x i1> %mask, <8 x i32> %a, <8 x i32> %passthru, i32 %vl)
  ret <8 x i32> %b
}

; Test float binary operator with vp.merge and vp.fadd.
declare <8 x float> @llvm.vp.fadd.nxv2f32(<8 x float>, <8 x float>, <8 x i1>, i32)
define <8 x float> @vpmerge_vpfadd(<8 x float> %passthru, <8 x float> %x, <8 x float> %y, <8 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpfadd:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfadd.vv v8, v9, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpfadd
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v9, $v10, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v10
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY4:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   [[PseudoVFADD_VV_M1_MASK:%[0-9]+]]:vrnov0 = nofpexcept PseudoVFADD_VV_M1_MASK [[COPY4]], [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0, implicit $frm
  ; MIR-NEXT:   $v8 = COPY [[PseudoVFADD_VV_M1_MASK]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x float> @llvm.vp.fadd.nxv2f32(<8 x float> %x, <8 x float> %y, <8 x i1> %mask, i32 %vl)
  %b = call <8 x float> @llvm.vp.merge.nxv2f32(<8 x i1> %m, <8 x float> %a, <8 x float> %passthru, i32 %vl)
  ret <8 x float> %b
}

; Test conversion by fptosi.
declare <8 x i16> @llvm.vp.fptosi.nxv2i16.nxv2f32(<8 x float>, <8 x i1>, i32)
define <8 x i16> @vpmerge_vpfptosi(<8 x i16> %passthru, <8 x float> %x, <8 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpfptosi:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, tu, mu
; CHECK-NEXT:    vfncvt.rtz.x.f.w v8, v9, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpfptosi
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v9, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   early-clobber %4:vrnov0 = nofpexcept PseudoVFNCVT_RTZ_X_F_W_MF2_MASK [[COPY3]], [[COPY2]], $v0, [[COPY]], 4 /* e16 */, 0
  ; MIR-NEXT:   $v8 = COPY %4
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x i16> @llvm.vp.fptosi.nxv2i16.nxv2f32(<8 x float> %x, <8 x i1> %mask, i32 %vl)
  %b = call <8 x i16> @llvm.vp.merge.nxv2i16(<8 x i1> %m, <8 x i16> %a, <8 x i16> %passthru, i32 %vl)
  ret <8 x i16> %b
}

; Test conversion by sitofp.
declare <8 x float> @llvm.vp.sitofp.nxv2f32.nxv2i64(<8 x i64>, <8 x i1>, i32)
define <8 x float> @vpmerge_vpsitofp(<8 x float> %passthru, <8 x i64> %x, <8 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpsitofp:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfncvt.f.x.w v8, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpsitofp
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v10m2, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vrm2 = COPY $v10m2
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   early-clobber %4:vrnov0 = nofpexcept PseudoVFNCVT_F_X_W_M1_MASK [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0, implicit $frm
  ; MIR-NEXT:   $v8 = COPY %4
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x float> @llvm.vp.sitofp.nxv2f32.nxv2i64(<8 x i64> %x, <8 x i1> %mask, i32 %vl)
  %b = call <8 x float> @llvm.vp.merge.nxv2f32(<8 x i1> %m, <8 x float> %a, <8 x float> %passthru, i32 %vl)
  ret <8 x float> %b
}

; Test integer extension by vp.zext.
declare <8 x i32> @llvm.vp.zext.nxv2i32.nxv2i8(<8 x i8>, <8 x i1>, i32)
define <8 x i32> @vpmerge_vpzext(<8 x i32> %passthru, <8 x i8> %x, <8 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpzext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vzext.vf4 v8, v9, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpzext
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v9, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   early-clobber %4:vrnov0 = PseudoVZEXT_VF4_M1_MASK [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY %4
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x i32> @llvm.vp.zext.nxv2i32.nxv2i8(<8 x i8> %x, <8 x i1> %mask, i32 %vl)
  %b = call <8 x i32> @llvm.vp.merge.nxv2i32(<8 x i1> %m, <8 x i32> %a, <8 x i32> %passthru, i32 %vl)
  ret <8 x i32> %b
}

; Test integer truncation by vp.trunc.
declare <8 x i32> @llvm.vp.trunc.nxv2i32.nxv2i64(<8 x i64>, <8 x i1>, i32)
define <8 x i32> @vpmerge_vptrunc(<8 x i32> %passthru, <8 x i64> %x, <8 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vptrunc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vnsrl.wi v8, v10, 0, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vptrunc
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v10m2, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vrm2 = COPY $v10m2
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   early-clobber %4:vrnov0 = PseudoVNSRL_WI_M1_MASK [[COPY3]], [[COPY2]], 0, $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY %4
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x i32> @llvm.vp.trunc.nxv2i32.nxv2i64(<8 x i64> %x, <8 x i1> %mask, i32 %vl)
  %b = call <8 x i32> @llvm.vp.merge.nxv2i32(<8 x i1> %m, <8 x i32> %a, <8 x i32> %passthru, i32 %vl)
  ret <8 x i32> %b
}

; Test integer extension by vp.fpext.
declare <8 x double> @llvm.vp.fpext.nxv2f64.nxv2f32(<8 x float>, <8 x i1>, i32)
define <8 x double> @vpmerge_vpfpext(<8 x double> %passthru, <8 x float> %x, <8 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpfpext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfwcvt.f.f.v v8, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpfpext
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8m2, $v10, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v10
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrm2nov0 = COPY $v8m2
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   early-clobber %4:vrm2nov0 = nofpexcept PseudoVFWCVT_F_F_V_M1_MASK [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8m2 = COPY %4
  ; MIR-NEXT:   PseudoRET implicit $v8m2
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x double> @llvm.vp.fpext.nxv2f64.nxv2f32(<8 x float> %x, <8 x i1> %mask, i32 %vl)
  %b = call <8 x double> @llvm.vp.merge.nxv2f64(<8 x i1> %m, <8 x double> %a, <8 x double> %passthru, i32 %vl)
  ret <8 x double> %b
}

; Test integer truncation by vp.trunc.
declare <8 x float> @llvm.vp.fptrunc.nxv2f32.nxv2f64(<8 x double>, <8 x i1>, i32)
define <8 x float> @vpmerge_vpfptrunc(<8 x float> %passthru, <8 x double> %x, <8 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpfptrunc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfncvt.f.f.w v8, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpfptrunc
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v10m2, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vrm2 = COPY $v10m2
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   early-clobber %4:vrnov0 = nofpexcept PseudoVFNCVT_F_F_W_M1_MASK [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0, implicit $frm
  ; MIR-NEXT:   $v8 = COPY %4
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x float> @llvm.vp.fptrunc.nxv2f32.nxv2f64(<8 x double> %x, <8 x i1> %mask, i32 %vl)
  %b = call <8 x float> @llvm.vp.merge.nxv2f32(<8 x i1> %m, <8 x float> %a, <8 x float> %passthru, i32 %vl)
  ret <8 x float> %b
}

; Test load operation by vp.load.
declare <8 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<8 x i32> *, <8 x i1>, i32)
define <8 x i32> @vpmerge_vpload(<8 x i32> %passthru, <8 x i32> * %p, <8 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpload:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a1, e32, m1, tu, mu
; CHECK-NEXT:    vle32.v v8, (a0), v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpload
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $x10, $v0, $x11
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x11
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:gpr = COPY $x10
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   [[PseudoVLE32_V_M1_MASK:%[0-9]+]]:vrnov0 = PseudoVLE32_V_M1_MASK [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY [[PseudoVLE32_V_M1_MASK]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<8 x i32> * %p, <8 x i1> %mask, i32 %vl)
  %b = call <8 x i32> @llvm.vp.merge.nxv2i32(<8 x i1> %m, <8 x i32> %a, <8 x i32> %passthru, i32 %vl)
  ret <8 x i32> %b
}

; Test result have chain and glued node.
define <8 x i32> @vpmerge_vpload2(<8 x i32> %passthru, <8 x i32> * %p, <8 x i32> %x, <8 x i32> %y, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpload2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a1, e32, m1, ta, mu
; CHECK-NEXT:    vmseq.vv v0, v9, v10
; CHECK-NEXT:    vsetvli zero, zero, e32, m1, tu, mu
; CHECK-NEXT:    vle32.v v8, (a0), v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpload2
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $x10, $v9, $v10, $x11
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x11
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v10
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:gpr = COPY $x10
  ; MIR-NEXT:   [[COPY4:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   [[PseudoVMSEQ_VV_M1_:%[0-9]+]]:vr = PseudoVMSEQ_VV_M1 [[COPY2]], [[COPY1]], [[COPY]], 5 /* e32 */
  ; MIR-NEXT:   $v0 = COPY [[PseudoVMSEQ_VV_M1_]]
  ; MIR-NEXT:   [[PseudoVLE32_V_M1_MASK:%[0-9]+]]:vrnov0 = PseudoVLE32_V_M1_MASK [[COPY4]], [[COPY3]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY [[PseudoVLE32_V_M1_MASK]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %splat = insertelement <8 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <8 x i1> %splat, <8 x i1> poison, <8 x i32> zeroinitializer
  %a = call <8 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<8 x i32> * %p, <8 x i1> %mask, i32 %vl)
  %m = call <8 x i1> @llvm.vp.icmp.nxv2i32(<8 x i32> %x, <8 x i32> %y, metadata !"eq", <8 x i1> %mask, i32 %vl)
  %b = call <8 x i32> @llvm.vp.merge.nxv2i32(<8 x i1> %m, <8 x i32> %a, <8 x i32> %passthru, i32 %vl)
  ret <8 x i32> %b
}
