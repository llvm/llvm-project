; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI,SICIVI,FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI,SICIVI,FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,FUNC %s


; FUNC-LABEL: {{^}}s_usubo_i64_zext:
; GCN: s_sub_u32
; GCN: s_subb_u32
; GCN: v_cmp_gt_u64_e32 vcc

; EG: SUBB_UINT
; EG: ADDC_UINT
define amdgpu_kernel void @s_usubo_i64_zext(i64 addrspace(1)* %out, i64 %a, i64 %b) #0 {
  %usub = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %a, i64 %b) #0
  %val = extractvalue { i64, i1 } %usub, 0
  %carry = extractvalue { i64, i1 } %usub, 1
  %ext = zext i1 %carry to i64
  %add2 = add i64 %val, %ext
  store i64 %add2, i64 addrspace(1)* %out, align 8
  ret void
}

; FIXME: Could do scalar

; FUNC-LABEL: {{^}}s_usubo_i32:
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}

; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc

; EG-DAG: SUBB_UINT
; EG-DAG: SUB_INT
define amdgpu_kernel void @s_usubo_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 %a, i32 %b) #0 {
  %usub = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %usub, 0
  %carry = extractvalue { i32, i1 } %usub, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_usubo_i32:
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}

; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc

; EG-DAG: SUBB_UINT
; EG-DAG: SUB_INT
define amdgpu_kernel void @v_usubo_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 addrspace(1)* %a.ptr, i32 addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i32, i32 addrspace(1)* %a.ptr
  %b.gep = getelementptr inbounds i32, i32 addrspace(1)* %b.ptr
  %a = load i32, i32 addrspace(1)* %a.gep, align 4
  %b = load i32, i32 addrspace(1)* %b.gep, align 4
  %usub = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %usub, 0
  %carry = extractvalue { i32, i1 } %usub, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_usubo_i32_novcc:
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}

; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc

; EG-DAG: SUBB_UINT
; EG-DAG: SUB_INT
define amdgpu_kernel void @v_usubo_i32_novcc(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 addrspace(1)* %a.ptr, i32 addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i32, i32 addrspace(1)* %a.ptr
  %b.gep = getelementptr inbounds i32, i32 addrspace(1)* %b.ptr
  %a = load i32, i32 addrspace(1)* %a.gep, align 4
  %b = load i32, i32 addrspace(1)* %b.gep, align 4
  %uadd = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %uadd, 0
  %carry = extractvalue { i32, i1 } %uadd, 1
  store volatile i32 %val, i32 addrspace(1)* %out, align 4
  call void asm sideeffect "", "~{vcc}"() #0
  store volatile i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}s_usubo_i64:
; GCN: s_sub_u32
; GCN: s_subb_u32

; EG-DAG: SUBB_UINT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG: SUB_INT
define amdgpu_kernel void @s_usubo_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 %a, i64 %b) #0 {
  %usub = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue { i64, i1 } %usub, 0
  %carry = extractvalue { i64, i1 } %usub, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_usubo_i64:
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_subb_u32
; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_subb_u32

; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_subb_co_u32

; EG-DAG: SUBB_UINT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG: SUB_INT
define amdgpu_kernel void @v_usubo_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 addrspace(1)* %a.ptr, i64 addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i64, i64 addrspace(1)* %a.ptr
  %b.gep = getelementptr inbounds i64, i64 addrspace(1)* %b.ptr
  %a = load i64, i64 addrspace(1)* %a.gep
  %b = load i64, i64 addrspace(1)* %b.gep
  %usub = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue { i64, i1 } %usub, 0
  %carry = extractvalue { i64, i1 } %usub, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_usubo_i16:
; SI: v_subrev_i32_e32
; SI: v_and_b32
; SI: v_cmp_ne_u32_e32

; VI: v_sub_u16_e32
; VI: v_cmp_gt_u16_e32

; GFX9: v_sub_u16_e32
; GFX9: v_cmp_gt_u16_e32
define amdgpu_kernel void @v_usubo_i16(i16 addrspace(1)* %out, i1 addrspace(1)* %carryout, i16 addrspace(1)* %a.ptr, i16 addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, i16 addrspace(1)* %a.ptr
  %b.gep = getelementptr inbounds i16, i16 addrspace(1)* %b.ptr
  %a = load i16, i16 addrspace(1)* %a.gep
  %b = load i16, i16 addrspace(1)* %b.gep
  %usub = call { i16, i1 } @llvm.usub.with.overflow.i16(i16 %a, i16 %b)
  %val = extractvalue { i16, i1 } %usub, 0
  %carry = extractvalue { i16, i1 } %usub, 1
  store i16 %val, i16 addrspace(1)* %out
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_usubo_v2i32:
; SICIVI: v_sub_{{[iu]}}32
; SICIVI: v_cndmask_b32
; SICIVI: v_sub_{{[iu]}}32
; SICIVI: v_cndmask_b32
define amdgpu_kernel void @v_usubo_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %carryout, <2 x i32> addrspace(1)* %aptr, <2 x i32> addrspace(1)* %bptr) nounwind {
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %aptr, align 4
  %b = load <2 x i32>, <2 x i32> addrspace(1)* %bptr, align 4
  %sadd = call { <2 x i32>, <2 x i1> } @llvm.usub.with.overflow.v2i32(<2 x i32> %a, <2 x i32> %b) nounwind
  %val = extractvalue { <2 x i32>, <2 x i1> } %sadd, 0
  %carry = extractvalue { <2 x i32>, <2 x i1> } %sadd, 1
  store <2 x i32> %val, <2 x i32> addrspace(1)* %out, align 4
  %carry.ext = zext <2 x i1> %carry to <2 x i32>
  store <2 x i32> %carry.ext, <2 x i32> addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}s_usubo_clamp_bit:
; GCN: v_sub_{{i|u|co_u}}32_e32
; GCN: s_endpgm
define amdgpu_kernel void @s_usubo_clamp_bit(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 %a, i32 %b) #0 {
entry:
  %usub = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %usub, 0
  %carry = extractvalue { i32, i1 } %usub, 1
  %c2 = icmp eq i1 %carry, false
  %cc = icmp eq i32 %a, %b
  br i1 %cc, label %exit, label %if

if:
  br label %exit

exit:
  %cout = phi i1 [false, %entry], [%c2, %if]
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %cout, i1 addrspace(1)* %carryout
  ret void
}


; FUNC-LABEL: {{^}}v_usubo_clamp_bit:
; GCN: v_sub_{{i|u|co_u}}32_e64
; GCN: s_endpgm
define amdgpu_kernel void @v_usubo_clamp_bit(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 addrspace(1)* %a.ptr, i32 addrspace(1)* %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i32, i32 addrspace(1)* %a.ptr
  %b.gep = getelementptr inbounds i32, i32 addrspace(1)* %b.ptr
  %a = load i32, i32 addrspace(1)* %a.gep, align 4
  %b = load i32, i32 addrspace(1)* %b.gep, align 4
  %usub = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %usub, 0
  %carry = extractvalue { i32, i1 } %usub, 1
  %c2 = icmp eq i1 %carry, false
  %cc = icmp eq i32 %a, %b
  br i1 %cc, label %exit, label %if

if:
  br label %exit

exit:
  %cout = phi i1 [false, %entry], [%c2, %if]
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %cout, i1 addrspace(1)* %carryout
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare { i16, i1 } @llvm.usub.with.overflow.i16(i16, i16) #1
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) #1
declare { i64, i1 } @llvm.usub.with.overflow.i64(i64, i64) #1
declare { <2 x i32>, <2 x i1> } @llvm.usub.with.overflow.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
