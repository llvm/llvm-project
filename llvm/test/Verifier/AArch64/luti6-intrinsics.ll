; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

define <vscale x 8 x i16> @bad_sve_luti6_ret(<vscale x 16 x i8> %a) {
; CHECK: Intrinsic has incorrect return type!
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.luti6(<vscale x 16 x i8> %a, <vscale x 16 x i8> %a, <vscale x 16 x i8> %a)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @bad_sve_luti6_lane_x2_arg(<vscale x 4 x i32> %a, <vscale x 16 x i8> %b) {
; CHECK: Intrinsic has incorrect argument type!
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.luti6.lane.x2.i16(<vscale x 4 x i32> %a, <vscale x 4 x i32> %a, <vscale x 16 x i8> %b, i32 1)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x half> @bad_sve_luti6_lane_x2_f16_arg(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK: Intrinsic has incorrect argument type!
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.luti6.lane.x2.f16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %a, <vscale x 16 x i8> %b, i32 1)
  ret <vscale x 8 x half> %res
}

define <vscale x 8 x i16> @bad_sme_luti6_zt_ret(i32 %zt, <vscale x 16 x i8> %idx) {
; CHECK: Intrinsic has incorrect return type!
  %res = call <vscale x 8 x i16> @llvm.aarch64.sme.luti6.zt(i32 %zt, <vscale x 16 x i8> %idx)
  ret <vscale x 8 x i16> %res
}

define { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @bad_sme_luti6_zt_x4_ret(i32 %zt, <vscale x 16 x i8> %a) {
; CHECK: Intrinsic has incorrect return type!
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sme.luti6.zt.x4(i32 %zt, <vscale x 16 x i8> %a, <vscale x 16 x i8> %a, <vscale x 16 x i8> %a)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @bad_sme_luti6_lane_x4_arg(<vscale x 8 x half> %a, <vscale x 16 x i8> %b) {
; CHECK: Intrinsic has incorrect argument type!
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sme.luti6.lane.x4.nxv8i16(<vscale x 8 x half> %a, <vscale x 8 x half> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %b, i32 1)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.luti6(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.luti6.lane.x2.i16(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x half> @llvm.aarch64.sve.luti6.lane.x2.f16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sme.luti6.zt(i32, <vscale x 16 x i8>)
declare { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sme.luti6.zt.x4(i32, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sme.luti6.lane.x4.nxv8i16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 16 x i8>, <vscale x 16 x i8>, i32)
