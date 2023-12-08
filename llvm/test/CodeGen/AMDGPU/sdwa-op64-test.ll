; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX9,GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefixes=FIJI,GCN %s

; GCN-LABEL: {{^}}test_add_co_sdwa:
; GFX9: v_add_co_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
; GFX9: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc{{$}}
; FIJI: v_add_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
; FIJI: v_addc_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc{{$}}
define void @test_add_co_sdwa(ptr addrspace(1) %arg, ptr addrspace(1) %arg1) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp3 = getelementptr inbounds i32, ptr addrspace(1) %arg1, i32 %tmp
  %tmp4 = load i32, ptr addrspace(1) %tmp3, align 4
  %tmp5 = and i32 %tmp4, 255
  %tmp6 = zext i32 %tmp5 to i64
  %tmp7 = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %tmp
  %tmp8 = load i64, ptr addrspace(1) %tmp7, align 8
  %tmp9 = add nsw i64 %tmp8, %tmp6
  store i64 %tmp9, ptr addrspace(1) %tmp7, align 8
  ret void
}


; GCN-LABEL: {{^}}test_sub_co_sdwa:
; GFX9: v_sub_co_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
; GFX9: v_subbrev_co_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc{{$}}
; FIJI: v_sub_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
; FIJI: v_subbrev_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc{{$}}
define void @test_sub_co_sdwa(ptr addrspace(1) %arg, ptr addrspace(1) %arg1) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp3 = getelementptr inbounds i32, ptr addrspace(1) %arg1, i32 %tmp
  %tmp4 = load i32, ptr addrspace(1) %tmp3, align 4
  %tmp5 = and i32 %tmp4, 255
  %tmp6 = zext i32 %tmp5 to i64
  %tmp7 = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %tmp
  %tmp8 = load i64, ptr addrspace(1) %tmp7, align 8
  %tmp9 = sub nsw i64 %tmp8, %tmp6
  store i64 %tmp9, ptr addrspace(1) %tmp7, align 8
  ret void
}

; GCN-LABEL: {{^}}test1_add_co_sdwa:
; GFX9: v_add_co_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
; GFX9: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc{{$}}
; GFX9: v_add_co_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
; GFX9: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc{{$}}
; FIJI: v_add_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
; FIJI: v_addc_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc{{$}}
; FIJI: v_add_u32_sdwa v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
; FIJI: v_addc_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc{{$}}
define amdgpu_kernel void @test1_add_co_sdwa(ptr addrspace(1) %arg, ptr addrspace(1) %arg1, ptr addrspace(1) %arg2) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp3 = getelementptr inbounds i32, ptr addrspace(1) %arg1, i32 %tmp
  %tmp4 = load i32, ptr addrspace(1) %tmp3, align 4
  %tmp5 = and i32 %tmp4, 255
  %tmp6 = zext i32 %tmp5 to i64
  %tmp7 = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %tmp
  %tmp8 = load i64, ptr addrspace(1) %tmp7, align 8
  %tmp9 = add nsw i64 %tmp8, %tmp6
  store i64 %tmp9, ptr addrspace(1) %tmp7, align 8
  %tmp13 = getelementptr inbounds i32, ptr addrspace(1) %arg1, i32 %tmp
  %tmp14 = load i32, ptr addrspace(1) %tmp13, align 4
  %tmp15 = and i32 %tmp14, 255
  %tmp16 = zext i32 %tmp15 to i64
  %tmp17 = getelementptr inbounds i64, ptr addrspace(1) %arg2, i32 %tmp
  %tmp18 = load i64, ptr addrspace(1) %tmp17, align 8
  %tmp19 = add nsw i64 %tmp18, %tmp16
  store i64 %tmp19, ptr addrspace(1) %tmp17, align 8
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
