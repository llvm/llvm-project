; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,CI %s
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI %s

; GCN-LABEL: {{^}}mad_i64_i32_sextops:
; CI: v_mad_i64_i32 v[0:1], s[6:7], v0, v1, v[2:3]

; SI: v_mul_lo_i32
; SI: v_mul_hi_i32
; SI: v_add_i32
; SI: v_addc_u32
define i64 @mad_i64_i32_sextops(i32 %arg0, i32 %arg1, i64 %arg2) #0 {
  %sext0 = sext i32 %arg0 to i64
  %sext1 = sext i32 %arg1 to i64
  %mul = mul i64 %sext0, %sext1
  %mad = add i64 %mul, %arg2
  ret i64 %mad
}

; GCN-LABEL: {{^}}mad_i64_i32_sextops_commute:
; CI: v_mad_i64_i32 v[0:1], s[6:7], v0, v1, v[2:3]

; SI-DAG: v_mul_lo_i32
; SI-DAG: v_mul_hi_i32
; SI: v_add_i32
; SI: v_addc_u32
define i64 @mad_i64_i32_sextops_commute(i32 %arg0, i32 %arg1, i64 %arg2) #0 {
  %sext0 = sext i32 %arg0 to i64
  %sext1 = sext i32 %arg1 to i64
  %mul = mul i64 %sext0, %sext1
  %mad = add i64 %arg2, %mul
  ret i64 %mad
}

; GCN-LABEL: {{^}}mad_u64_u32_zextops:
; CI: v_mad_u64_u32 v[0:1], s[6:7], v0, v1, v[2:3]

; SI-DAG: v_mul_lo_i32
; SI-DAG: v_mul_hi_u32
; SI: v_add_i32
; SI: v_addc_u32
define i64 @mad_u64_u32_zextops(i32 %arg0, i32 %arg1, i64 %arg2) #0 {
  %sext0 = zext i32 %arg0 to i64
  %sext1 = zext i32 %arg1 to i64
  %mul = mul i64 %sext0, %sext1
  %mad = add i64 %mul, %arg2
  ret i64 %mad
}

; GCN-LABEL: {{^}}mad_u64_u32_zextops_commute:
; CI: v_mad_u64_u32 v[0:1], s[6:7], v0, v1, v[2:3]

; SI-DAG: v_mul_lo_i32
; SI-DAG: v_mul_hi_u32
; SI: v_add_i32
; SI: v_addc_u32
define i64 @mad_u64_u32_zextops_commute(i32 %arg0, i32 %arg1, i64 %arg2) #0 {
  %sext0 = zext i32 %arg0 to i64
  %sext1 = zext i32 %arg1 to i64
  %mul = mul i64 %sext0, %sext1
  %mad = add i64 %arg2, %mul
  ret i64 %mad
}






; GCN-LABEL: {{^}}mad_i64_i32_sextops_i32_i128:
; CI: v_mad_u64_u32
; CI: v_mad_u64_u32
; CI: v_mad_i64_i32
; CI: v_mad_u64_u32


; SI-NOT: v_mad_
define i128 @mad_i64_i32_sextops_i32_i128(i32 %arg0, i32 %arg1, i128 %arg2) #0 {
  %sext0 = sext i32 %arg0 to i128
  %sext1 = sext i32 %arg1 to i128
  %mul = mul i128 %sext0, %sext1
  %mad = add i128 %mul, %arg2
  ret i128 %mad
}

; GCN-LABEL: {{^}}mad_i64_i32_sextops_i32_i63:
; CI: v_lshl_b64
; CI: v_ashr
; CI: v_mad_i64_i32 v[0:1], s[6:7], v0, v1, v[2:3]

; SI-NOT: v_mad_u64_u32
define i63 @mad_i64_i32_sextops_i32_i63(i32 %arg0, i32 %arg1, i63 %arg2) #0 {
  %sext0 = sext i32 %arg0 to i63
  %sext1 = sext i32 %arg1 to i63
  %mul = mul i63 %sext0, %sext1
  %mad = add i63 %mul, %arg2
  ret i63 %mad
}

; GCN-LABEL: {{^}}mad_i64_i32_sextops_i31_i63:
; CI: v_lshl_b64
; CI: v_bfe_i32 v[[B1:[0-9]+]], v1, 0, 31
; CI: v_ashr_i64
; CI: v_bfe_i32 v[[B2:[0-9]+]], v0, 0, 31
; CI: v_mad_i64_i32 v[0:1], s[6:7], v[[B2]], v[[B1]], v[1:2]
define i63 @mad_i64_i32_sextops_i31_i63(i31 %arg0, i31 %arg1, i63 %arg2) #0 {
  %sext0 = sext i31 %arg0 to i63
  %sext1 = sext i31 %arg1 to i63
  %mul = mul i63 %sext0, %sext1
  %mad = add i63 %mul, %arg2
  ret i63 %mad
}

; GCN-LABEL: {{^}}mad_u64_u32_bitops:
; CI: v_mad_u64_u32 v[0:1], s[6:7], v0, v2, v[4:5]
define i64 @mad_u64_u32_bitops(i64 %arg0, i64 %arg1, i64 %arg2) #0 {
  %trunc.lhs = and i64 %arg0, 4294967295
  %trunc.rhs = and i64 %arg1, 4294967295
  %mul = mul i64 %trunc.lhs, %trunc.rhs
  %add = add i64 %mul, %arg2
  ret i64 %add
}

; GCN-LABEL: {{^}}mad_u64_u32_bitops_lhs_mask_small:
; GCN-NOT: v_mad_
define i64 @mad_u64_u32_bitops_lhs_mask_small(i64 %arg0, i64 %arg1, i64 %arg2) #0 {
  %trunc.lhs = and i64 %arg0, 8589934591
  %trunc.rhs = and i64 %arg1, 4294967295
  %mul = mul i64 %trunc.lhs, %trunc.rhs
  %add = add i64 %mul, %arg2
  ret i64 %add
}

; GCN-LABEL: {{^}}mad_u64_u32_bitops_rhs_mask_small:
; GCN-NOT: v_mad_
define i64 @mad_u64_u32_bitops_rhs_mask_small(i64 %arg0, i64 %arg1, i64 %arg2) #0 {
  %trunc.lhs = and i64 %arg0, 4294967295
  %trunc.rhs = and i64 %arg1, 8589934591
  %mul = mul i64 %trunc.lhs, %trunc.rhs
  %add = add i64 %mul, %arg2
  ret i64 %add
}

; GCN-LABEL: {{^}}mad_i64_i32_bitops:
; CI: v_mad_i64_i32 v[0:1], s[6:7], v0, v2, v[4:5]
; SI-NOT: v_mad_
define i64 @mad_i64_i32_bitops(i64 %arg0, i64 %arg1, i64 %arg2) #0 {
  %shl.lhs = shl i64 %arg0, 32
  %trunc.lhs = ashr i64 %shl.lhs, 32
  %shl.rhs = shl i64 %arg1, 32
  %trunc.rhs = ashr i64 %shl.rhs, 32
  %mul = mul i64 %trunc.lhs, %trunc.rhs
  %add = add i64 %mul, %arg2
  ret i64 %add
}

; Example from bug report
; GCN-LABEL: {{^}}mad_i64_i32_unpack_i64ops:
; CI: v_mad_u64_u32 v[0:1], s[6:7], v1, v0, v[0:1]
; SI-NOT: v_mad_u64_u32
define i64 @mad_i64_i32_unpack_i64ops(i64 %arg0) #0 {
  %tmp4 = lshr i64 %arg0, 32
  %tmp5 = and i64 %arg0, 4294967295
  %mul = mul nuw i64 %tmp4, %tmp5
  %mad = add i64 %mul, %arg0
  ret i64 %mad
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
