; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Check that we do not use AGPRs for v32i32 type

; GCN-LABEL: {{^}}test_v1024:
; GCN-NOT: v_accvgpr
; GCN-COUNT-32: v_mov_b32_e32
; GCN-NOT: v_accvgpr
define amdgpu_kernel void @test_v1024() {
entry:
  %alloca = alloca <32 x i32>, align 16, addrspace(5)
  %cast = bitcast <32 x i32> addrspace(5)* %alloca to i8 addrspace(5)*
  call void @llvm.memset.p5i8.i32(i8 addrspace(5)* %cast, i8 0, i32 128, i1 false)
  br i1 undef, label %if.then.i.i, label %if.else.i

if.then.i.i:                                      ; preds = %entry
  call void @llvm.memcpy.p5i8.p5i8.i64(i8 addrspace(5)* align 16 %cast, i8 addrspace(5)* align 4 undef, i64 128, i1 false)
  br label %if.then.i62.i

if.else.i:                                        ; preds = %entry
  br label %if.then.i62.i

if.then.i62.i:                                    ; preds = %if.else.i, %if.then.i.i
  call void @llvm.memcpy.p1i8.p5i8.i64(i8 addrspace(1)* align 4 undef, i8 addrspace(5)* align 16 %cast, i64 128, i1 false)
  ret void
}

declare void @llvm.memset.p5i8.i32(i8 addrspace(5)* nocapture readonly, i8, i32, i1 immarg)
declare void @llvm.memcpy.p5i8.p5i8.i64(i8 addrspace(5)* nocapture writeonly, i8 addrspace(5)* nocapture readonly, i64, i1 immarg)

declare void @llvm.memcpy.p1i8.p5i8.i64(i8 addrspace(1)* nocapture writeonly, i8 addrspace(5)* nocapture readonly, i64, i1 immarg)
