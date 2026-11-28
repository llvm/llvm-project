; RUN: opt -S -passes=instcombine < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define <vscale x 4 x i32> @and_z_all_active(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) #0 {
; CHECK-LABEL: @and_z_all_active(
; CHECK-NEXT:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.and.u.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[A:%.*]], <vscale x 4 x i32> [[B:%.*]])
; CHECK-NEXT:    ret <vscale x 4 x i32> [[RES]]
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.and.z.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @orr_z_all_active(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) #0 {
; CHECK-LABEL: @orr_z_all_active(
; CHECK-NEXT:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.orr.u.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[A:%.*]], <vscale x 4 x i32> [[B:%.*]])
; CHECK-NEXT:    ret <vscale x 4 x i32> [[RES]]
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.orr.z.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @eor_z_all_active(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) #0 {
; CHECK-LABEL: @eor_z_all_active(
; CHECK-NEXT:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.eor.u.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[A:%.*]], <vscale x 4 x i32> [[B:%.*]])
; CHECK-NEXT:    ret <vscale x 4 x i32> [[RES]]
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.eor.z.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @and_z_predicated(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) #0 {
; CHECK-LABEL: @and_z_predicated(
; CHECK-NEXT:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.and.z.nxv4i32(<vscale x 4 x i1> [[PG:%.*]], <vscale x 4 x i32> [[A:%.*]], <vscale x 4 x i32> [[B:%.*]])
; CHECK-NEXT:    ret <vscale x 4 x i32> [[RES]]
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.and.z.nxv4i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

attributes #0 = { "target-features"="+sve" }
