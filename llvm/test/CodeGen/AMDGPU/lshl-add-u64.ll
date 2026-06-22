; RUN: llc -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck -check-prefixes=GCN,GFX942 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck -check-prefixes=GCN,GFX1250 %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck -check-prefixes=GISEL,GFX942-GISEL %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck -check-prefixes=GISEL,GFX1250-GISEL %s

define i64 @lshl_add_u64_v1v(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_v1v:
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 1, v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_v1v:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 1, v[{{[0-9:]+}}]
  %shl = shl i64 %v, 1
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_v4v(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_v4v:
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 4, v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_v4v:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 4, v[{{[0-9:]+}}]
  %shl = shl i64 %v, 4
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_v5v(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_v5v:
; GCN:      v_lshlrev_b64
; GFX942-NEXT: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 0, v[{{[0-9:]+}}]
; GFX1250-NEXT: s_delay_alu
; GFX1250-NEXT: v_add_nc_u64_e32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_v5v:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_add_nc_u64_e32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
  %shl = shl i64 %v, 5
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_vvv(i64 %v, i64 %s, i64 %a) {
; GCN-LABEL: lshl_add_u64_vvv:
; GCN:      v_lshlrev_b64
; GFX942-NEXT: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 0, v[{{[0-9:]+}}]
; GFX1250-NEXT: s_delay_alu
; GFX1250-NEXT: v_add_nc_u64_e32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_vvv:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_add_nc_u64_e32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
  %shl = shl i64 %v, %s
  %add = add i64 %shl, %a
  ret i64 %add
}

define amdgpu_kernel void @lshl_add_u64_s2v(i64 %v) {
; GCN-LABEL: lshl_add_u64_s2v:
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], s[{{[0-9:]+}}], 2, v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_s2v:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, s{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_add_nc_u64_e32 v[{{[0-9:]+}}], s[{{[0-9:]+}}], v[{{[0-9:]+}}]
  %a = load i64, ptr poison
  %shl = shl i64 %v, 2
  %add = add i64 %shl, %a
  store i64 %add, ptr poison
  ret void
}

define amdgpu_kernel void @lshl_add_u64_v2s(i64 %a) {
; GCN-LABEL: lshl_add_u64_v2s:
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 2, s[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_v2s:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, s{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 2, s[{{[0-9:]+}}]
  %v = load i64, ptr poison
  %shl = shl i64 %v, 2
  %add = add i64 %shl, %a
  store i64 %add, ptr poison
  ret void
}

define amdgpu_kernel void @lshl_add_u64_s2s(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_s2s:
; GCN:    s_lshl_b64
; GFX942: s_add_u32
; GFX942: s_addc_u32
; GFX1250: s_add_nc_u64
; GISEL-LABEL: lshl_add_u64_s2s:
; GFX942-GISEL: s_addc_u32
; GFX1250-GISEL: s_add_nc_u64
  %shl = shl i64 %v, 2
  %add = add i64 %shl, %a
  store i64 %add, ptr poison
  ret void
}

define i64 @add_u64_vv(i64 %v, i64 %a) {
; GCN-LABEL: add_u64_vv:
; GFX942: v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
; GFX1250: v_add_nc_u64_e32 v[0:1], v[0:1], v[2:3]
; GISEL-LABEL: add_u64_vv:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_add_nc_u64_e32 v[0:1], v[0:1], v[2:3]
  %add = add i64 %v, %a
  ret i64 %add
}

define amdgpu_kernel void @add_u64_sv(i64 %v) {
; GCN-LABEL: add_u64_sv:
; GFX942: v_lshl_add_u64 v[0:1], s[0:1], 0, v[0:1]
; GFX1250: v_add_nc_u64_e32 v[0:1], s[0:1], v[0:1]
; GISEL-LABEL: add_u64_sv:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, s{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_add_nc_u64_e32 v[0:1], s[0:1], v[0:1]
  %a = load i64, ptr poison
  %add = add i64 %v, %a
  store i64 %add, ptr poison
  ret void
}

define amdgpu_kernel void @add_u64_vs(i64 %a) {
; GCN-LABEL: add_u64_vs:
; GFX942: v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
; GFX1250: v_add_nc_u64_e32 v[0:1], s[0:1], v[0:1]
; GISEL-LABEL: add_u64_vs:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, s{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_add_nc_u64_e32 v[0:1], s[0:1], v[0:1]
  %v = load i64, ptr poison
  %add = add i64 %v, %a
  store i64 %add, ptr poison
  ret void
}

define amdgpu_kernel void @add_u64_ss(i64 %v, i64 %a) {
; GCN-LABEL: add_u64_ss:
; GFX942: s_add_u32
; GFX942: s_addc_u32 s1, s1, s3
; GFX1250: s_add_nc_u64 s[0:1], s[0:1], s[2:3]
; GISEL-LABEL: add_u64_ss:
; GFX1250-GISEL: s_add_nc_u64 s[0:1], s[0:1], s[2:3]
  %add = add i64 %v, %a
  store i64 %add, ptr poison
  ret void
}

define i32 @lshl_add_u64_gep(ptr %p, i64 %a) {
; GCN-LABEL: lshl_add_u64_gep:
; GCN: v_lshl_add_u64 v[0:1], v[2:3], 2, v[0:1]
; GISEL-LABEL: lshl_add_u64_gep:
; GISEL: v_lshl_add_u64 v[0:1], v[2:3], 2, v[0:1]
  %gep = getelementptr inbounds i32, ptr %p, i64 %a
  %v = load i32, ptr %gep
  ret i32 %v
}

define i64 @lshl_add_u64_vvv_and_2(i64 %v, i64 %a, i64 %s) {
; GCN-LABEL: lshl_add_u64_vvv_and_2:
; GCN: v_and_b32_e32 [[AND:v[0-9:]+]], 2, v{{[0-9:]+}}
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND]], v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_vvv_and_2:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_and_b32_e32 [[AND:v[0-9:]+]], 2, v{{[0-9:]+}}
; GFX1250-GISEL: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND]], v[{{[0-9:]+}}]
  %and = and i64 %s, 2
  %shl = shl i64 %v, %and
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_vvv_and_4(i64 %v, i64 %a, i64 %s) {
; GCN-LABEL: lshl_add_u64_vvv_and_4:
; GCN: v_and_b32_e32 [[AND:v[0-9:]+]], 4, v{{[0-9:]+}}
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND]], v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_vvv_and_4:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_and_b32_e32 [[AND:v[0-9:]+]], 4, v{{[0-9:]+}}
; GFX1250-GISEL: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND]], v[{{[0-9:]+}}]
  %and = and i64 %s, 4
  %shl = shl i64 %v, %and
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_vvv_and_5(i64 %v, i64 %a, i64 %s) {
; GCN-LABEL: lshl_add_u64_vvv_and_5:
; GFX942: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 0, v[{{[0-9:]+}}]
; GFX1250: v_add_nc_u64_e32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_vvv_and_5:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_add_nc_u64_e32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
  %and = and i64 %s, 5
  %shl = shl i64 %v, %and
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_vvv_urem(i64 %v, i64 %a, i64 %s) {
; GCN-LABEL: lshl_add_u64_vvv_urem:
; GCN: v_and_b32_e32 [[AND:v[0-9:]+]], 3, v{{[0-9:]+}}
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND]], v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_vvv_urem:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND]], v[{{[0-9:]+}}]
  %urem = urem i64 %s, 4
  %shl = shl i64 %v, %urem
  %add = add i64 %shl, %a
  ret i64 %add
}

define <4 x i64> @lshl_add_u64_vvv_and_2_v4(<4 x i64> %v, <4 x i64> %a, <4 x i64> %s) {
; GCN-LABEL: lshl_add_u64_vvv_and_2_v4:
; GCN-DAG: v_and_b32_e32 [[AND5:v[0-9:]+]], 5, v{{[0-9:]+}}
; GCN-DAG: v_and_b32_e32 [[AND3:v[0-9:]+]], 1, v{{[0-9:]+}}
; GCN-DAG: v_and_b32_e32 [[AND2:v[0-9:]+]], 2, v{{[0-9:]+}}
; GCN-DAG: v_and_b32_e32 [[AND1:v[0-9:]+]], 3, v{{[0-9:]+}}
; GCN-DAG: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND1]], v[{{[0-9:]+}}]
; GCN-DAG: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND2]], v[{{[0-9:]+}}]
; GCN-DAG: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND3]], v[{{[0-9:]+}}]
; GFX1250: v_add_nc_u64_e32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GFX942: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 0, v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_vvv_and_2_v4:
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX1250-GISEL-DAG: v_and_b32_e32 [[AND5:v[0-9:]+]], 5, v{{[0-9:]+}}
; GFX1250-GISEL-DAG: v_and_b32_e32 [[AND3:v[0-9:]+]], 1, v{{[0-9:]+}}
; GFX1250-GISEL-DAG: v_and_b32_e32 [[AND2:v[0-9:]+]], 2, v{{[0-9:]+}}
; GFX1250-GISEL-DAG: v_and_b32_e32 [[AND1:v[0-9:]+]], 3, v{{[0-9:]+}}
; GFX1250-GISEL-DAG: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND1]], v[{{[0-9:]+}}]
; GFX1250-GISEL-DAG v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[AND2]], v[{{[0-9:]+}}]
; GFX1250-GISEL: v_add_nc_u64_e32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
  %and = and <4 x i64> %s, <i64 1, i64 2, i64 3, i64 5>
  %shl = shl <4 x i64> %v, %and
  %add = add <4 x i64> %shl, %a
  ret <4 x i64> %add
}

define amdgpu_ps <2 x i32> @lshl_add_u64_sss_and_4(i32 inreg %v, i32 inreg %a, i32 inreg %s) {
; GCN-LABEL: lshl_add_u64_sss_and_4
; GFX942: s_add_i32 s{{[0-9:]+}}, s{{[0-9:]+}}, s{{[0-9:]+}}
; GFX1250: s_add_co_i32 s{{[0-9:]+}}, s{{[0-9:]+}}, s{{[0-9:]+}}
; GISEL-LABEL: lshl_add_u64_sss_and_4
; GFX942-GISEL: s_add_i32 s{{[0-9:]+}}, s{{[0-9:]+}}, s{{[0-9:]+}}
; GFX1250-GISEL: s_add_co_i32 s{{[0-9:]+}}, s{{[0-9:]+}}, s{{[0-9:]+}}
  %and = and i32 %s, 4
  %zext_and = zext i32 %and to i64
  %zext_a = zext i32 %and to i64
  %zext_v = zext i32 %and to i64
  %shl = shl i64 %zext_v, %zext_and
  %add = add i64 %shl, %zext_a
  %bitcast = bitcast i64 %add to <2 x i32>
  ret <2 x i32> %bitcast
}

define amdgpu_ps <2 x i32> @lshl_add_u64_svs_and_4(i32 inreg %v, i64 %a, i32 inreg %s) {
; GCN-LABEL: lshl_add_u64_svs_and_4
; GFX-1250: v_lshl_add_u64 v[{{[0-9:]+}}], s{{[0-9:]+}}, s{{[0-9:]+}}, v[{{[0-9:]+}}]
; GFX-942: v_lshl_add_u64 v[{{[0-9:]+}}], s[{{[0-9:]+}}], 0, v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_svs_and_4
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, s{{[0-9:]+}}, v{{[0-9:]+}}
; GFX-1250-GISEL: v_lshl_add_u64 v[{{[0-9:]+}}], s{{[0-9:]+}}], s{{[0-9:]+}}, v[{{[0-9:]+}}]
  %and = and i32 %s, 4
  %zext_and = zext i32 %and to i64
  %zext_v = zext i32 %and to i64
  %shl = shl i64 %zext_v, %zext_and
  %add = add i64 %shl, %a
  %bitcast = bitcast i64 %add to <2 x i32>
  ret <2 x i32> %bitcast
}

define amdgpu_ps <2 x i32> @lshl_add_u64_vvs_and_4(i64 %v, i64 %a, i32 inreg %s) {
; GCN-LABEL: lshl_add_u64_vvs_and_4
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s{{[0-9:]+}}, v[{{[0-9:]+}}]
; GISEL-LABEL: lshl_add_u64_vvs_and_4
; GFX942-GISEL: v_add_co_u32_e32 v{{[0-9:]+}}, vcc, v{{[0-9:]+}}, v{{[0-9:]+}}
; GFX-1250-GISEL: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s{{[0-9:]+}}, v[{{[0-9:]+}}]
  %and = and i32 %s, 4
  %zext_and = zext i32 %and to i64
  %shl = shl i64 %v, %zext_and
  %add = add i64 %shl, %a
  %bitcast = bitcast i64 %add to <2 x i32>
  ret <2 x i32> %bitcast
}
