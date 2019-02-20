; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mtriple=amdgcn-- -mattr=+promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
target datalayout = "A5"

declare {}* @llvm.invariant.start.p5i8(i64, i8 addrspace(5)* nocapture) #0
declare void @llvm.invariant.end.p5i8({}*, i64, i8 addrspace(5)* nocapture) #0
declare i8 addrspace(5)* @llvm.launder.invariant.group.p5i8(i8 addrspace(5)*) #1

; GCN-LABEL: {{^}}use_invariant_promotable_lds:
; GCN: buffer_load_dword
; GCN: ds_write_b32
define amdgpu_kernel void @use_invariant_promotable_lds(i32 addrspace(1)* %arg) #2 {
bb:
  %tmp = alloca i32, align 4, addrspace(5)
  %tmp1 = bitcast i32 addrspace(5)* %tmp to i8 addrspace(5)*
  %tmp2 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 1
  %tmp3 = load i32, i32 addrspace(1)* %tmp2
  store i32 %tmp3, i32 addrspace(5)* %tmp
  %tmp4 = call {}* @llvm.invariant.start.p5i8(i64 4, i8 addrspace(5)* %tmp1) #0
  call void @llvm.invariant.end.p5i8({}* %tmp4, i64 4, i8 addrspace(5)* %tmp1) #0
  %tmp5 = call i8 addrspace(5)* @llvm.launder.invariant.group.p5i8(i8 addrspace(5)* %tmp1) #1
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
