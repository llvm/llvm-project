; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI,GFX89 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,GFX89 %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() nounwind readnone

; GCN-LABEL: {{^}}anyext_i1_i32:
; GCN: v_cndmask_b32_e64
define amdgpu_kernel void @anyext_i1_i32(i32 addrspace(1)* %out, i32 %cond) #0 {
entry:
  %tmp = icmp eq i32 %cond, 0
  %tmp1 = zext i1 %tmp to i8
  %tmp2 = xor i8 %tmp1, -1
  %tmp3 = and i8 %tmp2, 1
  %tmp4 = zext i8 %tmp3 to i32
  store i32 %tmp4, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_anyext_i16_i32:
; GFX89: v_add_u16_e32 [[ADD:v[0-9]+]],
; GFX89: v_xor_b32_e32 [[XOR:v[0-9]+]], -1, [[ADD]]
; GFX89: v_and_b32_e32 [[AND:v[0-9]+]], 1, [[XOR]]
; GFX89: buffer_store_dword [[AND]]
define amdgpu_kernel void @s_anyext_i16_i32(i32 addrspace(1)* %out, i16 addrspace(1)* %a, i16 addrspace(1)* %b) #0 {
entry:
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.y = call i32 @llvm.amdgcn.workitem.id.y()
  %a.ptr = getelementptr i16, i16 addrspace(1)* %a, i32 %tid.x
  %b.ptr = getelementptr i16, i16 addrspace(1)* %b, i32 %tid.y
  %a.l = load i16, i16 addrspace(1)* %a.ptr
  %b.l = load i16, i16 addrspace(1)* %b.ptr
  %tmp = add i16 %a.l, %b.l
  %tmp1 = trunc i16 %tmp to i8
  %tmp2 = xor i8 %tmp1, -1
  %tmp3 = and i8 %tmp2, 1
  %tmp4 = zext i8 %tmp3 to i32
  store i32 %tmp4, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}anyext_v2i16_to_v2i32:
; GFX9: global_load_short_d16_hi
; GFX9: v_and_b32_e32 v{{[0-9]+}}, 0x80008000
; GFX9: v_bfi_b32 v{{[0-9]+}}, v{{[0-9]+}}, 0, v{{[0-9]+}}
; GFX9: v_cmp_eq_f32_e32
; GFX9: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
define amdgpu_kernel void @anyext_v2i16_to_v2i32() #0 {
bb:
  %tmp = load i16, i16 addrspace(1)* undef, align 2
  %tmp2 = insertelement <2 x i16> undef, i16 %tmp, i32 1
  %tmp4 = and <2 x i16> %tmp2, <i16 -32768, i16 -32768>
  %tmp5 = zext <2 x i16> %tmp4 to <2 x i32>
  %tmp6 = shl nuw <2 x i32> %tmp5, <i32 16, i32 16>
  %tmp7 = or <2 x i32> zeroinitializer, %tmp6
  %tmp8 = bitcast <2 x i32> %tmp7 to <2 x float>
  %tmp10 = fcmp oeq <2 x float> %tmp8, zeroinitializer
  %tmp11 = zext <2 x i1> %tmp10 to <2 x i8>
  %tmp12 = extractelement <2 x i8> %tmp11, i32 1
  store i8 %tmp12, i8 addrspace(1)* undef, align 1
  ret void
}

attributes #0 = { nounwind }
