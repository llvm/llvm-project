; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=kabini -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,16BANK %s
; RUN: llc -march=amdgcn -mcpu=stoney -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,16BANK %s

; GCN-LABEL: {{^}}v_interp:
; GCN-NOT: s_wqm
; GCN: s_mov_b32 m0, s{{[0-9]+}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.x{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.y{{$}}
; GCN-DAG: v_interp_p2_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.y{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, p0, attr0.x{{$}}
define amdgpu_ps void @v_interp(ptr addrspace(4) inreg %arg, ptr addrspace(4) inreg %arg1, ptr addrspace(4) inreg %arg2, i32 inreg %arg3, <2 x float> %arg4) #0 {
main_body:
  %i = extractelement <2 x float> %arg4, i32 0
  %j = extractelement <2 x float> %arg4, i32 1
  %p0_0 = call float @llvm.amdgcn.interp.p1(float %i, i32 0, i32 0, i32 %arg3)
  %p1_0 = call float @llvm.amdgcn.interp.p2(float %p0_0, float %j, i32 0, i32 0, i32 %arg3)
  %p0_1 = call float @llvm.amdgcn.interp.p1(float %i, i32 1, i32 0, i32 %arg3)
  %p1_1 = call float @llvm.amdgcn.interp.p2(float %p0_1, float %j, i32 1, i32 0, i32 %arg3)
  %const = call float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 %arg3)
  %w = fadd float %p1_1, %const
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %p0_0, float %p0_0, float %p1_1, float %w, i1 true, i1 true) #0
  ret void
}

; GCN-LABEL: {{^}}v_interp_p1:
; GCN: s_movk_i32 m0, 0x100
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.x{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.y{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.z{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.w{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.x{{$}}

; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr1.x{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr2.y{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr3.z{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr4.w{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr63.w{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr64.w{{$}}
; GCN-DAG: v_interp_p1_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr64.x{{$}}
define amdgpu_ps void @v_interp_p1(float %i) #0 {
bb:
  %p0_0 = call float @llvm.amdgcn.interp.p1(float %i, i32 0, i32 0, i32 256)
  %p0_1 = call float @llvm.amdgcn.interp.p1(float %i, i32 1, i32 0, i32 256)
  %p0_2 = call float @llvm.amdgcn.interp.p1(float %i, i32 2, i32 0, i32 256)
  %p0_3 = call float @llvm.amdgcn.interp.p1(float %i, i32 3, i32 0, i32 256)
  %p0_4 = call float @llvm.amdgcn.interp.p1(float %i, i32 4, i32 0, i32 256)
  %p0_5 = call float @llvm.amdgcn.interp.p1(float %i, i32 0, i32 1, i32 256)
  %p0_6 = call float @llvm.amdgcn.interp.p1(float %i, i32 1, i32 2, i32 256)
  %p0_7 = call float @llvm.amdgcn.interp.p1(float %i, i32 2, i32 3, i32 256)
  %p0_8 = call float @llvm.amdgcn.interp.p1(float %i, i32 3, i32 4, i32 256)
  %p0_9 = call float @llvm.amdgcn.interp.p1(float %i, i32 3, i32 63, i32 256)
  %p0_10 = call float @llvm.amdgcn.interp.p1(float %i, i32 3, i32 64, i32 256)
  %p0_11 = call float @llvm.amdgcn.interp.p1(float %i, i32 4, i32 64, i32 256)

  store volatile float %p0_0, ptr addrspace(1) undef
  store volatile float %p0_1, ptr addrspace(1) undef
  store volatile float %p0_2, ptr addrspace(1) undef
  store volatile float %p0_3, ptr addrspace(1) undef
  store volatile float %p0_4, ptr addrspace(1) undef
  store volatile float %p0_5, ptr addrspace(1) undef
  store volatile float %p0_6, ptr addrspace(1) undef
  store volatile float %p0_7, ptr addrspace(1) undef
  store volatile float %p0_8, ptr addrspace(1) undef
  store volatile float %p0_9, ptr addrspace(1) undef
  store volatile float %p0_10, ptr addrspace(1) undef
  store volatile float %p0_11, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}v_interp_p2:
; GCN: s_movk_i32 m0, 0x100
; GCN-DAG: v_interp_p2_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.x{{$}}
; GCN-DAG: v_interp_p2_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.y{{$}}
; GCN-DAG: v_interp_p2_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.z{{$}}
; GCN-DAG: v_interp_p2_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.w{{$}}
; GCN-DAG: v_interp_p2_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.x{{$}}
; GCN-DAG: v_interp_p2_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr0.x{{$}}
; GCN-DAG: v_interp_p2_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr63.x{{$}}
; GCN-DAG: v_interp_p2_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr64.x{{$}}
; GCN-DAG: v_interp_p2_f32{{(_e32)*}} v{{[0-9]+}}, v{{[0-9]+}}, attr64.x{{$}}
define amdgpu_ps void @v_interp_p2(float %x, float %j) #0 {
bb:
  %p2_0 = call float @llvm.amdgcn.interp.p2(float %x, float %j, i32 0, i32 0, i32 256)
  %p2_1 = call float @llvm.amdgcn.interp.p2(float %x, float %j, i32 1, i32 0, i32 256)
  %p2_2 = call float @llvm.amdgcn.interp.p2(float %x, float %j, i32 2, i32 0, i32 256)
  %p2_3 = call float @llvm.amdgcn.interp.p2(float %x, float %j, i32 3, i32 0, i32 256)
  %p2_4 = call float @llvm.amdgcn.interp.p2(float %x, float %j, i32 4, i32 0, i32 256)

  %p2_5 = call float @llvm.amdgcn.interp.p2(float %x, float %j, i32 0, i32 1, i32 256)
  %p2_6 = call float @llvm.amdgcn.interp.p2(float %x, float %j, i32 0, i32 63, i32 256)
  %p2_7 = call float @llvm.amdgcn.interp.p2(float %x, float %j, i32 0, i32 64, i32 256)
  %p2_8 = call float @llvm.amdgcn.interp.p2(float %x, float %j, i32 4, i32 64, i32 256)

  store volatile float %p2_0, ptr addrspace(1) undef
  store volatile float %p2_1, ptr addrspace(1) undef
  store volatile float %p2_2, ptr addrspace(1) undef
  store volatile float %p2_3, ptr addrspace(1) undef
  store volatile float %p2_4, ptr addrspace(1) undef
  store volatile float %p2_5, ptr addrspace(1) undef
  store volatile float %p2_6, ptr addrspace(1) undef
  store volatile float %p2_7, ptr addrspace(1) undef
  store volatile float %p2_8, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}v_interp_mov:
; GCN: s_movk_i32 m0, 0x100
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, p10, attr0.x{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, p20, attr0.x{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, p0, attr0.x{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, invalid_param_3, attr0.x{{$}}

; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, p10, attr0.x{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, p10, attr0.z{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, p10, attr0.w{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, p10, attr0.x{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, invalid_param_8, attr0.x{{$}}

; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, p10, attr63.y{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, p10, attr64.y{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, invalid_param_3, attr64.y{{$}}
; GCN-DAG: v_interp_mov_f32{{(_e32)*}} v{{[0-9]+}}, invalid_param_10, attr64.x{{$}}
define amdgpu_ps void @v_interp_mov(float %x, float %j) #0 {
bb:
  %mov_0 = call float @llvm.amdgcn.interp.mov(i32 0, i32 0, i32 0, i32 256)
  %mov_1 = call float @llvm.amdgcn.interp.mov(i32 1, i32 0, i32 0, i32 256)
  %mov_2 = call float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 256)
  %mov_3 = call float @llvm.amdgcn.interp.mov(i32 3, i32 0, i32 0, i32 256)

  %mov_4 = call float @llvm.amdgcn.interp.mov(i32 0, i32 1, i32 0, i32 256)
  %mov_5 = call float @llvm.amdgcn.interp.mov(i32 0, i32 2, i32 0, i32 256)
  %mov_6 = call float @llvm.amdgcn.interp.mov(i32 0, i32 3, i32 0, i32 256)
  %mov_7 = call float @llvm.amdgcn.interp.mov(i32 0, i32 4, i32 0, i32 256)
  %mov_8 = call float @llvm.amdgcn.interp.mov(i32 8, i32 4, i32 0, i32 256)

  %mov_9 = call float @llvm.amdgcn.interp.mov(i32 0, i32 1, i32 63, i32 256)
  %mov_10 = call float @llvm.amdgcn.interp.mov(i32 0, i32 1, i32 64, i32 256)
  %mov_11 = call float @llvm.amdgcn.interp.mov(i32 3, i32 1, i32 64, i32 256)
  %mov_12 = call float @llvm.amdgcn.interp.mov(i32 10, i32 4, i32 64, i32 256)

  store volatile float %mov_0, ptr addrspace(1) undef
  store volatile float %mov_1, ptr addrspace(1) undef
  store volatile float %mov_2, ptr addrspace(1) undef
  store volatile float %mov_3, ptr addrspace(1) undef

  store volatile float %mov_4, ptr addrspace(1) undef
  store volatile float %mov_5, ptr addrspace(1) undef
  store volatile float %mov_6, ptr addrspace(1) undef
  store volatile float %mov_7, ptr addrspace(1) undef
  store volatile float %mov_8, ptr addrspace(1) undef

  store volatile float %mov_9, ptr addrspace(1) undef
  store volatile float %mov_10, ptr addrspace(1) undef
  store volatile float %mov_11, ptr addrspace(1) undef
  store volatile float %mov_12, ptr addrspace(1) undef
  ret void
}

; SI won't merge ds memory operations, because of the signed offset bug, so
; we only have check lines for VI.
;
; TODO: VI won't merge them either, because we are conservative about moving
; instructions past changes to physregs.
;
; TODO-VI-LABEL: v_interp_readnone:
; TODO-VI: s_mov_b32 m0, 0
; TODO-VI-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; TODO-VI-DAG: v_interp_mov_f32_e32 v{{[0-9]+}}, p0, attr0.x{{$}}
; TODO-VI: s_mov_b32 m0, -1{{$}}
; TODO-VI: ds_write2_b32 v{{[0-9]+}}, [[ZERO]], [[ZERO]] offset1:4
;define amdgpu_ps void @v_interp_readnone(ptr addrspace(3) %lds) #0 {
;bb:
;  store float 0.000000e+00, ptr addrspace(3) %lds
;  %tmp1 = call float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 0)
;  %tmp2 = getelementptr float, ptr addrspace(3) %lds, i32 4
;  store float 0.000000e+00, ptr addrspace(3) %tmp2
;  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp1, float %tmp1, float %tmp1, float %tmp1, i1 true, i1 true) #0
;  ret void
;}

; Thest that v_interp_p1 uses different source and destination registers
; on 16 bank LDS chips.

; GCN-LABEL: {{^}}v_interp_p1_bank16_bug:
; 16BANK-NOT: v_interp_p1_f32{{(_e32)*}} [[DST:v[0-9]+]], [[DST]]
define amdgpu_ps void @v_interp_p1_bank16_bug(ptr addrspace(4) inreg %arg, ptr addrspace(4) inreg %arg13, ptr addrspace(4) inreg %arg14, ptr addrspace(4) inreg %arg15, float inreg %arg16, i32 inreg %arg17, <2 x i32> %arg18, <2 x i32> %arg19, <2 x i32> %arg20, <3 x i32> %arg21, <2 x i32> %arg22, <2 x i32> %arg23, <2 x i32> %arg24, float %arg25, float %arg26, float %arg27, float %arg28, float %arg29, float %arg30, i32 %arg31, float %arg32, float %arg33) #0 {
main_body:
  %i.i = extractelement <2 x i32> %arg19, i32 0
  %j.i = extractelement <2 x i32> %arg19, i32 1
  %i.f.i = bitcast i32 %i.i to float
  %j.f.i = bitcast i32 %j.i to float
  %p1.i = call float @llvm.amdgcn.interp.p1(float %i.f.i, i32 0, i32 0, i32 %arg17) #0
  %p2.i = call float @llvm.amdgcn.interp.p2(float %p1.i, float %j.f.i, i32 0, i32 0, i32 %arg17) #0
  %i.i7 = extractelement <2 x i32> %arg19, i32 0
  %j.i8 = extractelement <2 x i32> %arg19, i32 1
  %i.f.i9 = bitcast i32 %i.i7 to float
  %j.f.i10 = bitcast i32 %j.i8 to float
  %p1.i11 = call float @llvm.amdgcn.interp.p1(float %i.f.i9, i32 1, i32 0, i32 %arg17) #0
  %p2.i12 = call float @llvm.amdgcn.interp.p2(float %p1.i11, float %j.f.i10, i32 1, i32 0, i32 %arg17) #0
  %i.i1 = extractelement <2 x i32> %arg19, i32 0
  %j.i2 = extractelement <2 x i32> %arg19, i32 1
  %i.f.i3 = bitcast i32 %i.i1 to float
  %j.f.i4 = bitcast i32 %j.i2 to float
  %p1.i5 = call float @llvm.amdgcn.interp.p1(float %i.f.i3, i32 2, i32 0, i32 %arg17) #0
  %p2.i6 = call float @llvm.amdgcn.interp.p2(float %p1.i5, float %j.f.i4, i32 2, i32 0, i32 %arg17) #0
  %tmp = call float @llvm.fabs.f32(float %p2.i)
  %tmp34 = call float @llvm.fabs.f32(float %p2.i12)
  %tmp35 = call float @llvm.fabs.f32(float %p2.i6)
  %tmp36 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %tmp, float %tmp34)
  %tmp38 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %tmp35, float 1.000000e+00)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> %tmp36, <2 x half> %tmp38, i1 true, i1 true) #0
  ret void
}

declare float @llvm.fabs.f32(float) #1
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #1
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #1
declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32) #1
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1) #0
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
