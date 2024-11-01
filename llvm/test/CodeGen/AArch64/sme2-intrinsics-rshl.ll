; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme2 -verify-machineinstrs < %s | FileCheck %s

; SRSHL (Single, x2)

define { <vscale x 16 x i8>, <vscale x 16 x i8> } @multi_vec_rounding_shl_single_x2_s8(<vscale x 16 x i8> %dummy, <vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x2_s8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    srshl { z4.b, z5.b }, { z4.b, z5.b }, z3.b
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.srshl.single.x2.nxv16i8(<vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zm)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 8 x i16>, <vscale x 8 x i16> } @multi_vec_rounding_shl_single_x2_s16(<vscale x 8 x i16> %dummy, <vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x2_s16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    srshl { z4.h, z5.h }, { z4.h, z5.h }, z3.h
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.srshl.single.x2.nxv8i16(<vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zm)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 4 x i32>, <vscale x 4 x i32> } @multi_vec_rounding_shl_single_x2_s32(<vscale x 4 x i32> %dummy, <vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x2_s32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    srshl { z4.s, z5.s }, { z4.s, z5.s }, z3.s
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.srshl.single.x2.nxv4i32(<vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zm)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 2 x i64>, <vscale x 2 x i64> } @multi_vec_rounding_shl_single_x2_s64(<vscale x 2 x i64> %dummy, <vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x2_s64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    srshl { z4.d, z5.d }, { z4.d, z5.d }, z3.d
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.srshl.single.x2.nxv2i64(<vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zm)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

; SRSHL (Single, x4)

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }
@multi_vec_rounding_shl_single_x4_s8(<vscale x 16 x i8> %dummy, <vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zdn3, <vscale x 16 x i8> %zdn4, <vscale x 16 x i8> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x4_s8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    srshl { z24.b - z27.b }, { z24.b - z27.b }, z5.b
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }
              @llvm.aarch64.sve.srshl.single.x4.nxv16i8(<vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zdn3, <vscale x 16 x i8> %zdn4, <vscale x 16 x i8> %zm)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }
@multi_vec_rounding_shl_single_x4_s16(<vscale x 8 x i16> %dummy, <vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zdn3, <vscale x 8 x i16> %zdn4, <vscale x 8 x i16> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x4_s16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    srshl { z24.h - z27.h }, { z24.h - z27.h }, z5.h
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }
              @llvm.aarch64.sve.srshl.single.x4.nxv8i16(<vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zdn3, <vscale x 8 x i16> %zdn4, <vscale x 8 x i16> %zm)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }
@multi_vec_rounding_shl_single_x4_s32(<vscale x 4 x i32> %dummy, <vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zdn3, <vscale x 4 x i32> %zdn4, <vscale x 4 x i32> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x4_s32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    srshl { z24.s - z27.s }, { z24.s - z27.s }, z5.s
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }
              @llvm.aarch64.sve.srshl.single.x4.nxv4i32(<vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zdn3, <vscale x 4 x i32> %zdn4, <vscale x 4 x i32> %zm)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> }
@multi_vec_rounding_shl_single_x4_s64(<vscale x 2 x i64> %dummy, <vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zdn3, <vscale x 2 x i64> %zdn4, <vscale x 2 x i64> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x4_s64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    srshl { z24.d - z27.d }, { z24.d - z27.d }, z5.d
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> }
              @llvm.aarch64.sve.srshl.single.x4.nxv2i64(<vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zdn3, <vscale x 2 x i64> %zdn4, <vscale x 2 x i64> %zm)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

; URSHL (Single, x2)

define { <vscale x 16 x i8>, <vscale x 16 x i8> } @multi_vec_rounding_shl_single_x2_u8(<vscale x 16 x i8> %dummy, <vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x2_u8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    urshl { z4.b, z5.b }, { z4.b, z5.b }, z3.b
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.urshl.single.x2.nxv16i8(<vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zm)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 8 x i16>, <vscale x 8 x i16> } @multi_vec_rounding_shl_single_x2_u16(<vscale x 8 x i16> %dummy, <vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x2_u16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    urshl { z4.h, z5.h }, { z4.h, z5.h }, z3.h
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.urshl.single.x2.nxv8i16(<vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zm)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 4 x i32>, <vscale x 4 x i32> } @multi_vec_rounding_shl_single_x2_u32(<vscale x 4 x i32> %dummy, <vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x2_u32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    urshl { z4.s, z5.s }, { z4.s, z5.s }, z3.s
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.urshl.single.x2.nxv4i32(<vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zm)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 2 x i64>, <vscale x 2 x i64> } @multi_vec_rounding_shl_single_x2_u64(<vscale x 2 x i64> %dummy, <vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x2_u64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    urshl { z4.d, z5.d }, { z4.d, z5.d }, z3.d
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.urshl.single.x2.nxv2i64(<vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zm)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

; URSHL (Single, x4)

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }
@multi_vec_rounding_shl_single_x4_u8(<vscale x 16 x i8> %dummy, <vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zdn3, <vscale x 16 x i8> %zdn4, <vscale x 16 x i8> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x4_u8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    urshl { z24.b - z27.b }, { z24.b - z27.b }, z5.b
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }
              @llvm.aarch64.sve.urshl.single.x4.nxv16i8(<vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zdn3, <vscale x 16 x i8> %zdn4, <vscale x 16 x i8> %zm)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }
@multi_vec_rounding_shl_single_x4_u16(<vscale x 8 x i16> %dummy, <vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zdn3, <vscale x 8 x i16> %zdn4, <vscale x 8 x i16> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x4_u16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    urshl { z24.h - z27.h }, { z24.h - z27.h }, z5.h
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }
              @llvm.aarch64.sve.urshl.single.x4.nxv8i16(<vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zdn3, <vscale x 8 x i16> %zdn4, <vscale x 8 x i16> %zm)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }
@multi_vec_rounding_shl_single_x4_u32(<vscale x 4 x i32> %dummy, <vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zdn3, <vscale x 4 x i32> %zdn4, <vscale x 4 x i32> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x4_u32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    urshl { z24.s - z27.s }, { z24.s - z27.s }, z5.s
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }
              @llvm.aarch64.sve.urshl.single.x4.nxv4i32(<vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zdn3, <vscale x 4 x i32> %zdn4, <vscale x 4 x i32> %zm)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> }
@multi_vec_rounding_shl_single_x4_u64(<vscale x 2 x i64> %dummy, <vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zdn3, <vscale x 2 x i64> %zdn4, <vscale x 2 x i64> %zm) {
; CHECK-LABEL: multi_vec_rounding_shl_single_x4_u64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    urshl { z24.d - z27.d }, { z24.d - z27.d }, z5.d
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> }
              @llvm.aarch64.sve.urshl.single.x4.nxv2i64(<vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zdn3, <vscale x 2 x i64> %zdn4, <vscale x 2 x i64> %zm)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

; SRSHL (Multi, x2)

define { <vscale x 16 x i8>, <vscale x 16 x i8> } @multi_vec_rounding_shl_x2_s8(<vscale x 16 x i8> %dummy, <vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zm1, <vscale x 16 x i8> %zm2) {
; CHECK-LABEL: multi_vec_rounding_shl_x2_s8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z7.d, z4.d
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z6.d, z3.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    srshl { z4.b, z5.b }, { z4.b, z5.b }, { z6.b, z7.b }
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.srshl.x2.nxv16i8(<vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zm1, <vscale x 16 x i8> %zm2)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 8 x i16>, <vscale x 8 x i16> } @multi_vec_rounding_shl_x2_s16(<vscale x 8 x i16> %dummy, <vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zm1, <vscale x 8 x i16> %zm2) {
; CHECK-LABEL: multi_vec_rounding_shl_x2_s16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z7.d, z4.d
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z6.d, z3.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    srshl { z4.h, z5.h }, { z4.h, z5.h }, { z6.h, z7.h }
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.srshl.x2.nxv8i16(<vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zm1, <vscale x 8 x i16> %zm2)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 4 x i32>, <vscale x 4 x i32> } @multi_vec_rounding_shl_x2_s32(<vscale x 4 x i32> %dummy, <vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zm1, <vscale x 4 x i32> %zm2) {
; CHECK-LABEL: multi_vec_rounding_shl_x2_s32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z7.d, z4.d
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z6.d, z3.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    srshl { z4.s, z5.s }, { z4.s, z5.s }, { z6.s, z7.s }
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.srshl.x2.nxv4i32(<vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zm1, <vscale x 4 x i32> %zm2)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 2 x i64>, <vscale x 2 x i64> } @multi_vec_rounding_shl_x2_s64(<vscale x 2 x i64> %dummy, <vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zm1, <vscale x 2 x i64> %zm2) {
; CHECK-LABEL: multi_vec_rounding_shl_x2_s64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z7.d, z4.d
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z6.d, z3.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    srshl { z4.d, z5.d }, { z4.d, z5.d }, { z6.d, z7.d }
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.srshl.x2.nxv2i64(<vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zm1, <vscale x 2 x i64> %zm2)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

; SRSHL (Multi, x4)

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }
@multi_vec_rounding_shl_x4_s8(<vscale x 16 x i8> %dummy, <vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zdn3, <vscale x 16 x i8> %zdn4, <vscale x 16 x i8> %zm1, <vscale x 16 x i8> %zm2, <vscale x 16 x i8> %zm3, <vscale x 16 x i8> %zm4) {
; CHECK-LABEL: multi_vec_rounding_shl_x4_s8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z30.d, z7.d
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    mov z29.d, z6.d
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z28.d, z5.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    ld1b { z31.b }, p0/z, [x0]
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    srshl { z24.b - z27.b }, { z24.b - z27.b }, { z28.b - z31.b }
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }
              @llvm.aarch64.sve.srshl.x4.nxv16i8(<vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zdn3, <vscale x 16 x i8> %zdn4,
                                                 <vscale x 16 x i8> %zm1, <vscale x 16 x i8> %zm2, <vscale x 16 x i8> %zm3, <vscale x 16 x i8> %zm4)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }
@multi_vec_rounding_shl_x4_s16(<vscale x 8 x i16> %dummy, <vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zdn3, <vscale x 8 x i16> %zdn4, <vscale x 8 x i16> %zm1, <vscale x 8 x i16> %zm2, <vscale x 8 x i16> %zm3, <vscale x 8 x i16> %zm4) {
; CHECK-LABEL: multi_vec_rounding_shl_x4_s16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z30.d, z7.d
; CHECK-NEXT:    ptrue p0.h
; CHECK-NEXT:    mov z29.d, z6.d
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z28.d, z5.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    ld1h { z31.h }, p0/z, [x0]
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    srshl { z24.h - z27.h }, { z24.h - z27.h }, { z28.h - z31.h }
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }
              @llvm.aarch64.sve.srshl.x4.nxv8i16(<vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zdn3, <vscale x 8 x i16> %zdn4,
                                                 <vscale x 8 x i16> %zm1, <vscale x 8 x i16> %zm2, <vscale x 8 x i16> %zm3, <vscale x 8 x i16> %zm4)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }
@multi_vec_rounding_shl_x4_s32(<vscale x 4 x i32> %dummy, <vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zdn3, <vscale x 4 x i32> %zdn4, <vscale x 4 x i32> %zm1, <vscale x 4 x i32> %zm2, <vscale x 4 x i32> %zm3, <vscale x 4 x i32> %zm4) {
; CHECK-LABEL: multi_vec_rounding_shl_x4_s32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z30.d, z7.d
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    mov z29.d, z6.d
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z28.d, z5.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    ld1w { z31.s }, p0/z, [x0]
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    srshl { z24.s - z27.s }, { z24.s - z27.s }, { z28.s - z31.s }
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }
              @llvm.aarch64.sve.srshl.x4.nxv4i32(<vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zdn3, <vscale x 4 x i32> %zdn4,
                                                 <vscale x 4 x i32> %zm1, <vscale x 4 x i32> %zm2, <vscale x 4 x i32> %zm3, <vscale x 4 x i32> %zm4)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> }
@multi_vec_rounding_shl_x4_s64(<vscale x 2 x i64> %dummy, <vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zdn3, <vscale x 2 x i64> %zdn4, <vscale x 2 x i64> %zm1, <vscale x 2 x i64> %zm2, <vscale x 2 x i64> %zm3, <vscale x 2 x i64> %zm4) {
; CHECK-LABEL: multi_vec_rounding_shl_x4_s64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z30.d, z7.d
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    mov z29.d, z6.d
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z28.d, z5.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    ld1d { z31.d }, p0/z, [x0]
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    srshl { z24.d - z27.d }, { z24.d - z27.d }, { z28.d - z31.d }
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> }
              @llvm.aarch64.sve.srshl.x4.nxv2i64(<vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zdn3, <vscale x 2 x i64> %zdn4,
                                                 <vscale x 2 x i64> %zm1, <vscale x 2 x i64> %zm2, <vscale x 2 x i64> %zm3, <vscale x 2 x i64> %zm4)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

; URSHL (Multi, x2)

define { <vscale x 16 x i8>, <vscale x 16 x i8> } @multi_vec_rounding_uhl_x2_u8(<vscale x 16 x i8> %dummy, <vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zm1, <vscale x 16 x i8> %zm2) {
; CHECK-LABEL: multi_vec_rounding_uhl_x2_u8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z7.d, z4.d
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z6.d, z3.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    urshl { z4.b, z5.b }, { z4.b, z5.b }, { z6.b, z7.b }
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.urshl.x2.nxv16i8(<vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zm1, <vscale x 16 x i8> %zm2)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 8 x i16>, <vscale x 8 x i16> } @multi_vec_rounding_uhl_x2_u16(<vscale x 8 x i16> %dummy, <vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zm1, <vscale x 8 x i16> %zm2) {
; CHECK-LABEL: multi_vec_rounding_uhl_x2_u16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z7.d, z4.d
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z6.d, z3.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    urshl { z4.h, z5.h }, { z4.h, z5.h }, { z6.h, z7.h }
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.urshl.x2.nxv8i16(<vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zm1, <vscale x 8 x i16> %zm2)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 4 x i32>, <vscale x 4 x i32> } @multi_vec_rounding_uhl_x2_u32(<vscale x 4 x i32> %dummy, <vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zm1, <vscale x 4 x i32> %zm2) {
; CHECK-LABEL: multi_vec_rounding_uhl_x2_u32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z7.d, z4.d
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z6.d, z3.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    urshl { z4.s, z5.s }, { z4.s, z5.s }, { z6.s, z7.s }
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.urshl.x2.nxv4i32(<vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zm1, <vscale x 4 x i32> %zm2)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 2 x i64>, <vscale x 2 x i64> } @multi_vec_rounding_uhl_x2_u64(<vscale x 2 x i64> %dummy, <vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zm1, <vscale x 2 x i64> %zm2) {
; CHECK-LABEL: multi_vec_rounding_uhl_x2_u64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z7.d, z4.d
; CHECK-NEXT:    mov z5.d, z2.d
; CHECK-NEXT:    mov z6.d, z3.d
; CHECK-NEXT:    mov z4.d, z1.d
; CHECK-NEXT:    urshl { z4.d, z5.d }, { z4.d, z5.d }, { z6.d, z7.d }
; CHECK-NEXT:    mov z0.d, z4.d
; CHECK-NEXT:    mov z1.d, z5.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.urshl.x2.nxv2i64(<vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zm1, <vscale x 2 x i64> %zm2)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

; URSHL (Multi, x4)

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }
@multi_vec_rounding_shl_x4_u8(<vscale x 16 x i8> %dummy, <vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zdn3, <vscale x 16 x i8> %zdn4, <vscale x 16 x i8> %zm1, <vscale x 16 x i8> %zm2, <vscale x 16 x i8> %zm3, <vscale x 16 x i8> %zm4) {
; CHECK-LABEL: multi_vec_rounding_shl_x4_u8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z30.d, z7.d
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    mov z29.d, z6.d
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z28.d, z5.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    ld1b { z31.b }, p0/z, [x0]
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    urshl { z24.b - z27.b }, { z24.b - z27.b }, { z28.b - z31.b }
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }
              @llvm.aarch64.sve.urshl.x4.nxv16i8(<vscale x 16 x i8> %zdn1, <vscale x 16 x i8> %zdn2, <vscale x 16 x i8> %zdn3, <vscale x 16 x i8> %zdn4,
                                                 <vscale x 16 x i8> %zm1, <vscale x 16 x i8> %zm2, <vscale x 16 x i8> %zm3, <vscale x 16 x i8> %zm4)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }
@multi_vec_rounding_shl_x4_u16(<vscale x 8 x i16> %dummy, <vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zdn3, <vscale x 8 x i16> %zdn4, <vscale x 8 x i16> %zm1, <vscale x 8 x i16> %zm2, <vscale x 8 x i16> %zm3, <vscale x 8 x i16> %zm4) {
; CHECK-LABEL: multi_vec_rounding_shl_x4_u16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z30.d, z7.d
; CHECK-NEXT:    ptrue p0.h
; CHECK-NEXT:    mov z29.d, z6.d
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z28.d, z5.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    ld1h { z31.h }, p0/z, [x0]
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    urshl { z24.h - z27.h }, { z24.h - z27.h }, { z28.h - z31.h }
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }
              @llvm.aarch64.sve.urshl.x4.nxv8i16(<vscale x 8 x i16> %zdn1, <vscale x 8 x i16> %zdn2, <vscale x 8 x i16> %zdn3, <vscale x 8 x i16> %zdn4,
                                                 <vscale x 8 x i16> %zm1, <vscale x 8 x i16> %zm2, <vscale x 8 x i16> %zm3, <vscale x 8 x i16> %zm4)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }
@multi_vec_rounding_shl_x4_u32(<vscale x 4 x i32> %dummy, <vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zdn3, <vscale x 4 x i32> %zdn4, <vscale x 4 x i32> %zm1, <vscale x 4 x i32> %zm2, <vscale x 4 x i32> %zm3, <vscale x 4 x i32> %zm4) {
; CHECK-LABEL: multi_vec_rounding_shl_x4_u32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z30.d, z7.d
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    mov z29.d, z6.d
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z28.d, z5.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    ld1w { z31.s }, p0/z, [x0]
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    urshl { z24.s - z27.s }, { z24.s - z27.s }, { z28.s - z31.s }
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }
              @llvm.aarch64.sve.urshl.x4.nxv4i32(<vscale x 4 x i32> %zdn1, <vscale x 4 x i32> %zdn2, <vscale x 4 x i32> %zdn3, <vscale x 4 x i32> %zdn4,
                                                 <vscale x 4 x i32> %zm1, <vscale x 4 x i32> %zm2, <vscale x 4 x i32> %zm3, <vscale x 4 x i32> %zm4)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> }
@multi_vec_rounding_shl_x4_u64(<vscale x 2 x i64> %dummy, <vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zdn3, <vscale x 2 x i64> %zdn4, <vscale x 2 x i64> %zm1, <vscale x 2 x i64> %zm2, <vscale x 2 x i64> %zm3, <vscale x 2 x i64> %zm4) {
; CHECK-LABEL: multi_vec_rounding_shl_x4_u64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z30.d, z7.d
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    mov z29.d, z6.d
; CHECK-NEXT:    mov z27.d, z4.d
; CHECK-NEXT:    mov z28.d, z5.d
; CHECK-NEXT:    mov z26.d, z3.d
; CHECK-NEXT:    ld1d { z31.d }, p0/z, [x0]
; CHECK-NEXT:    mov z25.d, z2.d
; CHECK-NEXT:    mov z24.d, z1.d
; CHECK-NEXT:    urshl { z24.d - z27.d }, { z24.d - z27.d }, { z28.d - z31.d }
; CHECK-NEXT:    mov z0.d, z24.d
; CHECK-NEXT:    mov z1.d, z25.d
; CHECK-NEXT:    mov z2.d, z26.d
; CHECK-NEXT:    mov z3.d, z27.d
; CHECK-NEXT:    ret
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> }
              @llvm.aarch64.sve.urshl.x4.nxv2i64(<vscale x 2 x i64> %zdn1, <vscale x 2 x i64> %zdn2, <vscale x 2 x i64> %zdn3, <vscale x 2 x i64> %zdn4,
                                                 <vscale x 2 x i64> %zm1, <vscale x 2 x i64> %zm2, <vscale x 2 x i64> %zm3, <vscale x 2 x i64> %zm4)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

declare { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.srshl.single.x2.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.srshl.single.x2.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.srshl.single.x2.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.srshl.single.x2.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.srshl.single.x4.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.srshl.single.x4.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.srshl.single.x4.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.srshl.single.x4.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.urshl.single.x2.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.urshl.single.x2.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.urshl.single.x2.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.urshl.single.x2.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.urshl.single.x4.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.urshl.single.x4.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.urshl.single.x4.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.urshl.single.x4.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.srshl.x2.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.srshl.x2.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>,  <vscale x 8 x i16>)
declare { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.srshl.x2.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.srshl.x2.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }
 @llvm.aarch64.sve.srshl.x4.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }
 @llvm.aarch64.sve.srshl.x4.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }
 @llvm.aarch64.sve.srshl.x4.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> }
 @llvm.aarch64.sve.srshl.x4.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> )

declare { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.urshl.x2.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.urshl.x2.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.urshl.x2.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.urshl.x2.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }
 @llvm.aarch64.sve.urshl.x4.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }
 @llvm.aarch64.sve.urshl.x4.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }
 @llvm.aarch64.sve.urshl.x4.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> }
 @llvm.aarch64.sve.urshl.x4.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)
