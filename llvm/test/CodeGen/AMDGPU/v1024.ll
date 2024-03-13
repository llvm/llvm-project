; RUN: llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Check that we do not use AGPRs for v32i32 type

; GCN-LABEL: {{^}}test_v1024:
; GCN-NOT: v_accvgpr
; GCN-COUNT-8: global_store_dwordx4
; GCN-NOT: v_accvgpr
define amdgpu_kernel void @test_v1024(i1 %c0) {
entry:
  %alloca = alloca <32 x i32>, align 16, addrspace(5)
  call void @llvm.memset.p5.i32(ptr addrspace(5) %alloca, i8 0, i32 128, i1 false)
  br i1 %c0, label %if.then.i.i, label %if.else.i

if.then.i.i:                                      ; preds = %entry
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 16 %alloca, ptr addrspace(5) align 4 undef, i64 128, i1 false)
  br label %if.then.i62.i

if.else.i:                                        ; preds = %entry
  br label %if.then.i62.i

if.then.i62.i:                                    ; preds = %if.else.i, %if.then.i.i
  call void @llvm.memcpy.p1.p5.i64(ptr addrspace(1) align 4 undef, ptr addrspace(5) align 16 %alloca, i64 128, i1 false)
  ret void
}

declare void @llvm.memset.p5.i32(ptr addrspace(5) nocapture readonly, i8, i32, i1 immarg)
declare void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg)

declare void @llvm.memcpy.p1.p5.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg)
