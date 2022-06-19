; RUN: llc -march=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: name:            s_ctlz_i32
; GCN: S_FLBIT_I32_B32
define amdgpu_kernel void @s_ctlz_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 false) nounwind readnone
  store i32 %ctlz, i32 addrspace(1)* %out, align 4
  ret void
}
; GCN-LABEL: name:            v_ctlz_i32
; GCN: V_FFBH_U32_e64
define amdgpu_kernel void @v_ctlz_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i32, i32 addrspace(1)* %valptr, i32 %tid
  %val = load i32, i32 addrspace(1)* %in.gep, align 4
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 false) nounwind readnone
  store i32 %ctlz, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            s_cttz_i32
; GCN: S_FF1_I32_B32
define amdgpu_kernel void @s_cttz_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {	
  %cttz = call i32 @llvm.cttz.i32(i32 %val, i1 false) nounwind readnone
  store i32 %cttz, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            v_cttz_i32
; GCN: V_FFBL_B32_e64
define amdgpu_kernel void @v_cttz_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i32, i32 addrspace(1)* %valptr, i32 %tid
  %val = load i32, i32 addrspace(1)* %in.gep, align 4
  %cttz = call i32 @llvm.cttz.i32(i32 %val, i1 false) nounwind readnone
  store i32 %cttz, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            s_flbit
; GCN: S_FLBIT_I32
define amdgpu_kernel void @s_flbit(i32 addrspace(1)* noalias %out, i32 %val) #0 {
  %r = call i32 @llvm.amdgcn.sffbh.i32(i32 %val)
  store i32 %r, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            v_flbit
; GCN: V_FFBH_I32_e64
define amdgpu_kernel void @v_flbit(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i32, i32 addrspace(1)* %valptr, i32 %tid
  %val = load i32, i32 addrspace(1)* %in.gep, align 4
  %r = call i32 @llvm.amdgcn.sffbh.i32(i32 %val)
  store i32 %r, i32 addrspace(1)* %out, align 4
  ret void
}


declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone
declare i32 @llvm.cttz.i32(i32, i1) nounwind readnone
declare i32 @llvm.amdgcn.sffbh.i32(i32)
declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

