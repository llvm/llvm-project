; RUN: llc < %s -mtriple=riscv64 -mattr=+v | FileCheck %s
; RUN: llc < %s -mtriple=riscv64 -mattr=+v -stop-after=finalize-isel | FileCheck %s --check-prefix=MIR

declare <vscale x 2 x i16> @llvm.vp.merge.nxv2i16(<vscale x 2 x i1>, <vscale x 2 x i16>, <vscale x 2 x i16>, i32)
declare <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i32)
declare <vscale x 2 x float> @llvm.vp.merge.nxv2f32(<vscale x 2 x i1>, <vscale x 2 x float>, <vscale x 2 x float>, i32)
declare <vscale x 2 x double> @llvm.vp.merge.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, i32)

; Test binary operator with vp.merge and vp.smax.
declare <vscale x 2 x i32> @llvm.vp.add.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i32)
define <vscale x 2 x i32> @vpmerge_vpadd(<vscale x 2 x i32> %passthru, <vscale x 2 x i32> %x, <vscale x 2 x i32> %y, <vscale x 2 x i1> %m, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x i32> @llvm.vp.add.nxv2i32(<vscale x 2 x i32> %x, <vscale x 2 x i32> %y, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test glued node of merge should not be deleted.
declare <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, metadata, <vscale x 2 x i1>, i32)
define <vscale x 2 x i32> @vpmerge_vpadd2(<vscale x 2 x i32> %passthru, <vscale x 2 x i32> %x, <vscale x 2 x i32> %y, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x i32> @llvm.vp.add.nxv2i32(<vscale x 2 x i32> %x, <vscale x 2 x i32> %y, <vscale x 2 x i1> %mask, i32 %vl)
  %m = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %x, <vscale x 2 x i32> %y, metadata !"eq", <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test vp.merge has all-ones mask.
define <vscale x 2 x i32> @vpmerge_vpadd3(<vscale x 2 x i32> %passthru, <vscale x 2 x i32> %x, <vscale x 2 x i32> %y, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x i32> @llvm.vp.add.nxv2i32(<vscale x 2 x i32> %x, <vscale x 2 x i32> %y, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %mask, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test float binary operator with vp.merge and vp.fadd.
declare <vscale x 2 x float> @llvm.vp.fadd.nxv2f32(<vscale x 2 x float>, <vscale x 2 x float>, <vscale x 2 x i1>, i32)
define <vscale x 2 x float> @vpmerge_vpfadd(<vscale x 2 x float> %passthru, <vscale x 2 x float> %x, <vscale x 2 x float> %y, <vscale x 2 x i1> %m, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x float> @llvm.vp.fadd.nxv2f32(<vscale x 2 x float> %x, <vscale x 2 x float> %y, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x float> @llvm.vp.merge.nxv2f32(<vscale x 2 x i1> %m, <vscale x 2 x float> %a, <vscale x 2 x float> %passthru, i32 %vl)
  ret <vscale x 2 x float> %b
}

; Test for binary operator with specific EEW by riscv.vrgatherei16.
declare <vscale x 2 x i32> @llvm.riscv.vrgatherei16.vv.nxv2i32.i64(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i16>, i64)
define <vscale x 2 x i32> @vpmerge_vrgatherei16(<vscale x 2 x i32> %passthru, <vscale x 2 x i32> %x, <vscale x 2 x i16> %y, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vrgatherei16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vrgatherei16.vv v8, v9, v10
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vrgatherei16
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v9, $v10, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v10
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   early-clobber %5:vr = PseudoVRGATHEREI16_VV_M1_MF2 [[COPY2]], [[COPY1]], [[COPY]], 5 /* e32 */
  ; MIR-NEXT:   $v8 = COPY %5
  ; MIR-NEXT:   PseudoRET implicit $v8
  %1 = zext i32 %vl to i64
  %2 = tail call <vscale x 2 x i32> @llvm.riscv.vrgatherei16.vv.nxv2i32.i64(<vscale x 2 x i32> undef, <vscale x 2 x i32> %x, <vscale x 2 x i16> %y, i64 %1)
  %3 = tail call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %2, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %2
}

; Test conversion by fptosi.
declare <vscale x 2 x i16> @llvm.vp.fptosi.nxv2i16.nxv2f32(<vscale x 2 x float>, <vscale x 2 x i1>, i32)
define <vscale x 2 x i16> @vpmerge_vpfptosi(<vscale x 2 x i16> %passthru, <vscale x 2 x float> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x i16> @llvm.vp.fptosi.nxv2i16.nxv2f32(<vscale x 2 x float> %x, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x i16> @llvm.vp.merge.nxv2i16(<vscale x 2 x i1> %m, <vscale x 2 x i16> %a, <vscale x 2 x i16> %passthru, i32 %vl)
  ret <vscale x 2 x i16> %b
}

; Test conversion by sitofp.
declare <vscale x 2 x float> @llvm.vp.sitofp.nxv2f32.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i32)
define <vscale x 2 x float> @vpmerge_vpsitofp(<vscale x 2 x float> %passthru, <vscale x 2 x i64> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x float> @llvm.vp.sitofp.nxv2f32.nxv2i64(<vscale x 2 x i64> %x, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x float> @llvm.vp.merge.nxv2f32(<vscale x 2 x i1> %m, <vscale x 2 x float> %a, <vscale x 2 x float> %passthru, i32 %vl)
  ret <vscale x 2 x float> %b
}

; Test integer extension by vp.zext.
declare <vscale x 2 x i32> @llvm.vp.zext.nxv2i32.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i1>, i32)
define <vscale x 2 x i32> @vpmerge_vpzext(<vscale x 2 x i32> %passthru, <vscale x 2 x i8> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x i32> @llvm.vp.zext.nxv2i32.nxv2i8(<vscale x 2 x i8> %x, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test integer truncation by vp.trunc.
declare <vscale x 2 x i32> @llvm.vp.trunc.nxv2i32.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i32)
define <vscale x 2 x i32> @vpmerge_vptrunc(<vscale x 2 x i32> %passthru, <vscale x 2 x i64> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x i32> @llvm.vp.trunc.nxv2i32.nxv2i64(<vscale x 2 x i64> %x, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test integer extension by vp.fpext.
declare <vscale x 2 x double> @llvm.vp.fpext.nxv2f64.nxv2f32(<vscale x 2 x float>, <vscale x 2 x i1>, i32)
define <vscale x 2 x double> @vpmerge_vpfpext(<vscale x 2 x double> %passthru, <vscale x 2 x float> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x double> @llvm.vp.fpext.nxv2f64.nxv2f32(<vscale x 2 x float> %x, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x double> @llvm.vp.merge.nxv2f64(<vscale x 2 x i1> %m, <vscale x 2 x double> %a, <vscale x 2 x double> %passthru, i32 %vl)
  ret <vscale x 2 x double> %b
}

; Test integer truncation by vp.trunc.
declare <vscale x 2 x float> @llvm.vp.fptrunc.nxv2f32.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, i32)
define <vscale x 2 x float> @vpmerge_vpfptrunc(<vscale x 2 x float> %passthru, <vscale x 2 x double> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x float> @llvm.vp.fptrunc.nxv2f32.nxv2f64(<vscale x 2 x double> %x, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x float> @llvm.vp.merge.nxv2f32(<vscale x 2 x i1> %m, <vscale x 2 x float> %a, <vscale x 2 x float> %passthru, i32 %vl)
  ret <vscale x 2 x float> %b
}

; Test load operation by vp.load.
declare <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32> *, <vscale x 2 x i1>, i32)
define <vscale x 2 x i32> @vpmerge_vpload(<vscale x 2 x i32> %passthru, <vscale x 2 x i32> * %p, <vscale x 2 x i1> %m, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32> * %p, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test result has chain and glued node.
define <vscale x 2 x i32> @vpmerge_vpload2(<vscale x 2 x i32> %passthru, <vscale x 2 x i32> * %p, <vscale x 2 x i32> %x, <vscale x 2 x i32> %y, i32 zeroext %vl) {
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
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32> * %p, <vscale x 2 x i1> %mask, i32 %vl)
  %m = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %x, <vscale x 2 x i32> %y, metadata !"eq", <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test result has chain output of true operand of merge.vvm.
define void @vpmerge_vpload_store(<vscale x 2 x i32> %passthru, <vscale x 2 x i32> * %p, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vpload_store:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a1, e32, m1, tu, mu
; CHECK-NEXT:    vle32.v v8, (a0), v0.t
; CHECK-NEXT:    vs1r.v v8, (a0)
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vpload_store
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $x10, $v0, $x11
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x11
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:gpr = COPY $x10
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   [[PseudoVLE32_V_M1_MASK:%[0-9]+]]:vrnov0 = PseudoVLE32_V_M1_MASK [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   VS1R_V killed [[PseudoVLE32_V_M1_MASK]], [[COPY2]] :: (store unknown-size into %ir.p, align 8)
  ; MIR-NEXT:   PseudoRET
  %splat = insertelement <vscale x 2 x i1> poison, i1 -1, i32 0
  %mask = shufflevector <vscale x 2 x i1> %splat, <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer
  %a = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32> * %p, <vscale x 2 x i1> %mask, i32 %vl)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  store <vscale x 2 x i32> %b, <vscale x 2 x i32> * %p
  ret void
}

; FIXME: Merge vmerge.vvm and vleffN.v
declare { <vscale x 2 x i32>, i64 } @llvm.riscv.vleff.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>*, i64)
define <vscale x 2 x i32> @vpmerge_vleff(<vscale x 2 x i32> %passthru, <vscale x 2 x i32> * %p, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vleff:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a1, e32, m1, ta, mu
; CHECK-NEXT:    vle32ff.v v9, (a0)
; CHECK-NEXT:    vsetvli zero, a1, e32, m1, tu, mu
; CHECK-NEXT:    vmerge.vvm v8, v8, v9, v0
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vleff
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $x10, $v0, $x11
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x11
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:gpr = COPY $x10
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   [[PseudoVLE32FF_V_M1_:%[0-9]+]]:vr, [[PseudoVLE32FF_V_M1_1:%[0-9]+]]:gpr = PseudoVLE32FF_V_M1 [[COPY2]], [[COPY]], 5 /* e32 */, implicit-def dead $vl
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   [[PseudoVMERGE_VVM_M1_TU:%[0-9]+]]:vrnov0 = PseudoVMERGE_VVM_M1_TU [[COPY3]], [[COPY3]], killed [[PseudoVLE32FF_V_M1_]], $v0, [[COPY]], 5 /* e32 */
  ; MIR-NEXT:   $v8 = COPY [[PseudoVMERGE_VVM_M1_TU]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %1 = zext i32 %vl to i64
  %a = call { <vscale x 2 x i32>, i64 } @llvm.riscv.vleff.nxv2i32(<vscale x 2 x i32> undef, <vscale x 2 x i32>* %p, i64 %1)
  %b = extractvalue { <vscale x 2 x i32>, i64 } %a, 0
  %c = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %b, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %c
}

; Test strided load by riscv.vlse
declare <vscale x 2 x i32> @llvm.riscv.vlse.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>*, i64, i64)
define <vscale x 2 x i32> @vpmerge_vlse(<vscale x 2 x i32> %passthru,  <vscale x 2 x i32> * %p, <vscale x 2 x i1> %m, i64 %s, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vlse:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a2, e32, m1, tu, mu
; CHECK-NEXT:    vlse32.v v8, (a0), a1, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vlse
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $x10, $v0, $x11, $x12
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x12
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:gpr = COPY $x11
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:gpr = COPY $x10
  ; MIR-NEXT:   [[COPY4:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY2]]
  ; MIR-NEXT:   [[PseudoVLSE32_V_M1_MASK:%[0-9]+]]:vrnov0 = PseudoVLSE32_V_M1_MASK [[COPY4]], [[COPY3]], [[COPY1]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY [[PseudoVLSE32_V_M1_MASK]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %1 = zext i32 %vl to i64
  %a = call <vscale x 2 x i32> @llvm.riscv.vlse.nxv2i32(<vscale x 2 x i32> undef, <vscale x 2 x i32>* %p, i64 %s, i64 %1)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test indexed load by riscv.vluxei
declare <vscale x 2 x i32> @llvm.riscv.vluxei.nxv2i32.nxv2i64(<vscale x 2 x i32>, <vscale x 2 x i32>*, <vscale x 2 x i64>, i64)
define <vscale x 2 x i32> @vpmerge_vluxei(<vscale x 2 x i32> %passthru,  <vscale x 2 x i32> * %p, <vscale x 2 x i64> %idx, <vscale x 2 x i1> %m, i64 %s, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vluxei:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a2, e32, m1, tu, mu
; CHECK-NEXT:    vluxei64.v v8, (a0), v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vluxei
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $x10, $v10m2, $v0, $x12
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x12
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vrm2 = COPY $v10m2
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:gpr = COPY $x10
  ; MIR-NEXT:   [[COPY4:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   early-clobber %6:vrnov0 = PseudoVLUXEI64_V_M2_M1_MASK [[COPY4]], [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY %6
  ; MIR-NEXT:   PseudoRET implicit $v8
  %1 = zext i32 %vl to i64
  %a = call <vscale x 2 x i32> @llvm.riscv.vluxei.nxv2i32.nxv2i64(<vscale x 2 x i32> undef, <vscale x 2 x i32>* %p, <vscale x 2 x i64> %idx, i64 %1)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test vector index by riscv.vid
declare <vscale x 2 x i32> @llvm.riscv.vid.nxv2i32(<vscale x 2 x i32>, i64)
define <vscale x 2 x i32> @vpmerge_vid(<vscale x 2 x i32> %passthru, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vid:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vid.v v8, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vid
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   [[PseudoVID_V_M1_MASK:%[0-9]+]]:vrnov0 = PseudoVID_V_M1_MASK [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY [[PseudoVID_V_M1_MASK]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %1 = zext i32 %vl to i64
  %a = call <vscale x 2 x i32> @llvm.riscv.vid.nxv2i32(<vscale x 2 x i32> undef, i64 %1)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test riscv.viota
declare <vscale x 2 x i32> @llvm.riscv.viota.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i64)
define <vscale x 2 x i32> @vpmerge_viota(<vscale x 2 x i32> %passthru, <vscale x 2 x i1> %m, <vscale x 2 x i1> %vm, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_viota:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    viota.m v8, v9, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_viota
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v0, $v9, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY2]]
  ; MIR-NEXT:   early-clobber %4:vrnov0 = PseudoVIOTA_M_M1_MASK [[COPY3]], [[COPY1]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY %4
  ; MIR-NEXT:   PseudoRET implicit $v8
  %1 = zext i32 %vl to i64
  %a = call <vscale x 2 x i32> @llvm.riscv.viota.nxv2i32(<vscale x 2 x i32> undef, <vscale x 2 x i1> %vm, i64 %1)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test riscv.vfclass
declare <vscale x 2 x i32> @llvm.riscv.vfclass.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x float>, i64)
define <vscale x 2 x i32> @vpmerge_vflcass(<vscale x 2 x i32> %passthru, <vscale x 2 x float> %vf, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vflcass:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfclass.v v8, v9, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vflcass
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v9, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   [[PseudoVFCLASS_V_M1_MASK:%[0-9]+]]:vrnov0 = PseudoVFCLASS_V_M1_MASK [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0
  ; MIR-NEXT:   $v8 = COPY [[PseudoVFCLASS_V_M1_MASK]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %1 = zext i32 %vl to i64
  %a = call <vscale x 2 x i32> @llvm.riscv.vfclass.nxv2i32(<vscale x 2 x i32> undef, <vscale x 2 x float> %vf, i64 %1)
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test riscv.vfsqrt
declare <vscale x 2 x float> @llvm.riscv.vfsqrt.nxv2f32(<vscale x 2 x float>, <vscale x 2 x float>, i64)
define <vscale x 2 x float> @vpmerge_vfsqrt(<vscale x 2 x float> %passthru, <vscale x 2 x float> %vf, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vfsqrt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfsqrt.v v8, v9, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vfsqrt
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v9, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   [[PseudoVFSQRT_V_M1_MASK:%[0-9]+]]:vrnov0 = nofpexcept PseudoVFSQRT_V_M1_MASK [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0, implicit $frm
  ; MIR-NEXT:   $v8 = COPY [[PseudoVFSQRT_V_M1_MASK]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %1 = zext i32 %vl to i64
  %a = call <vscale x 2 x float> @llvm.riscv.vfsqrt.nxv2f32(<vscale x 2 x float> undef, <vscale x 2 x float> %vf, i64 %1)
  %b = call <vscale x 2 x float> @llvm.vp.merge.nxv2f32(<vscale x 2 x i1> %m, <vscale x 2 x float> %a, <vscale x 2 x float> %passthru, i32 %vl)
  ret <vscale x 2 x float> %b
}

; Test reciprocal operation by riscv.vfrec7
declare <vscale x 2 x float> @llvm.riscv.vfrec7.nxv2f32(<vscale x 2 x float>, <vscale x 2 x float>, i64)
define <vscale x 2 x float> @vpmerge_vfrec7(<vscale x 2 x float> %passthru, <vscale x 2 x float> %vf, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_vfrec7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfrec7.v v8, v9, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_vfrec7
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v9, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   [[PseudoVFREC7_V_M1_MASK:%[0-9]+]]:vrnov0 = nofpexcept PseudoVFREC7_V_M1_MASK [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0, implicit $frm
  ; MIR-NEXT:   $v8 = COPY [[PseudoVFREC7_V_M1_MASK]]
  ; MIR-NEXT:   PseudoRET implicit $v8
  %1 = zext i32 %vl to i64
  %a = call <vscale x 2 x float> @llvm.riscv.vfrec7.nxv2f32(<vscale x 2 x float> undef, <vscale x 2 x float> %vf, i64 %1)
  %b = call <vscale x 2 x float> @llvm.vp.merge.nxv2f32(<vscale x 2 x i1> %m, <vscale x 2 x float> %a, <vscale x 2 x float> %passthru, i32 %vl)
  ret <vscale x 2 x float> %b
}

; Test vector operations with VLMAX vector length.

; Test binary operator with vp.merge and add.
define <vscale x 2 x i32> @vpmerge_add(<vscale x 2 x i32> %passthru, <vscale x 2 x i32> %x, <vscale x 2 x i32> %y, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_add:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vadd.vv v8, v9, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_add
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
  %a = add <vscale x 2 x i32> %x, %y
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test binary operator with vp.merge and fadd.
define <vscale x 2 x float> @vpmerge_fadd(<vscale x 2 x float> %passthru, <vscale x 2 x float> %x, <vscale x 2 x float> %y, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_fadd:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfadd.vv v8, v9, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_fadd
  ; MIR: bb.0 (%ir-block.0):
  ; MIR-NEXT:   liveins: $v8, $v9, $v10, $v0, $x10
  ; MIR-NEXT: {{  $}}
  ; MIR-NEXT:   [[COPY:%[0-9]+]]:gprnox0 = COPY $x10
  ; MIR-NEXT:   [[COPY1:%[0-9]+]]:vr = COPY $v0
  ; MIR-NEXT:   [[COPY2:%[0-9]+]]:vr = COPY $v10
  ; MIR-NEXT:   [[COPY3:%[0-9]+]]:vr = COPY $v9
  ; MIR-NEXT:   [[COPY4:%[0-9]+]]:vrnov0 = COPY $v8
  ; MIR-NEXT:   $v0 = COPY [[COPY1]]
  ; MIR-NEXT:   %5:vrnov0 = nofpexcept PseudoVFADD_VV_M1_MASK [[COPY4]], [[COPY3]], [[COPY2]], $v0, [[COPY]], 5 /* e32 */, 0, implicit $frm
  ; MIR-NEXT:   $v8 = COPY %5
  ; MIR-NEXT:   PseudoRET implicit $v8
  %a = fadd <vscale x 2 x float> %x, %y
  %b = call <vscale x 2 x float> @llvm.vp.merge.nxv2f32(<vscale x 2 x i1> %m, <vscale x 2 x float> %a, <vscale x 2 x float> %passthru, i32 %vl)
  ret <vscale x 2 x float> %b
}

; Test conversion by fptosi.
define <vscale x 2 x i16> @vpmerge_fptosi(<vscale x 2 x i16> %passthru, <vscale x 2 x float> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_fptosi:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, tu, mu
; CHECK-NEXT:    vfncvt.rtz.x.f.w v8, v9, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_fptosi
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
  %a = fptosi <vscale x 2 x float> %x to <vscale x 2 x i16>
  %b = call <vscale x 2 x i16> @llvm.vp.merge.nxv2i16(<vscale x 2 x i1> %m, <vscale x 2 x i16> %a, <vscale x 2 x i16> %passthru, i32 %vl)
  ret <vscale x 2 x i16> %b
}

; Test conversion by sitofp.
define <vscale x 2 x float> @vpmerge_sitofp(<vscale x 2 x float> %passthru, <vscale x 2 x i64> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_sitofp:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfncvt.f.x.w v8, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_sitofp
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
  %a = sitofp <vscale x 2 x i64> %x to <vscale x 2 x float>
  %b = call <vscale x 2 x float> @llvm.vp.merge.nxv2f32(<vscale x 2 x i1> %m, <vscale x 2 x float> %a, <vscale x 2 x float> %passthru, i32 %vl)
  ret <vscale x 2 x float> %b
}

; Test float extension by fpext.
define <vscale x 2 x double> @vpmerge_fpext(<vscale x 2 x double> %passthru, <vscale x 2 x float> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_fpext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfwcvt.f.f.v v8, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_fpext
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
  %a = fpext <vscale x 2 x float> %x to <vscale x 2 x double>
  %b = call <vscale x 2 x double> @llvm.vp.merge.nxv2f64(<vscale x 2 x i1> %m, <vscale x 2 x double> %a, <vscale x 2 x double> %passthru, i32 %vl)
  ret <vscale x 2 x double> %b
}

; Test float truncation by fptrunc.
define <vscale x 2 x float> @vpmerge_fptrunc(<vscale x 2 x float> %passthru, <vscale x 2 x double> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_fptrunc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfncvt.f.f.w v8, v10, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_fptrunc
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
  %a = fptrunc <vscale x 2 x double> %x to <vscale x 2 x float>
  %b = call <vscale x 2 x float> @llvm.vp.merge.nxv2f32(<vscale x 2 x i1> %m, <vscale x 2 x float> %a, <vscale x 2 x float> %passthru, i32 %vl)
  ret <vscale x 2 x float> %b
}

; Test integer extension by zext.
define <vscale x 2 x i32> @vpmerge_zext(<vscale x 2 x i32> %passthru, <vscale x 2 x i8> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_zext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vzext.vf4 v8, v9, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_zext
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
  %a = zext <vscale x 2 x i8> %x to <vscale x 2 x i32>
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}

; Test integer truncation by trunc.
define <vscale x 2 x i32> @vpmerge_trunc(<vscale x 2 x i32> %passthru, <vscale x 2 x i64> %x, <vscale x 2 x i1> %m, i32 zeroext %vl) {
; CHECK-LABEL: vpmerge_trunc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vnsrl.wi v8, v10, 0, v0.t
; CHECK-NEXT:    ret
  ; MIR-LABEL: name: vpmerge_trunc
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
  %a = trunc <vscale x 2 x i64> %x to <vscale x 2 x i32>
  %b = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %passthru, i32 %vl)
  ret <vscale x 2 x i32> %b
}
