; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}zext_shl64_to_32:
; GCN: s_lshl_b32
; GCN-NOT: s_lshl_b64
define amdgpu_kernel void @zext_shl64_to_32(i64 addrspace(1)* nocapture %out, i32 %x) {
  %and = and i32 %x, 1073741823
  %ext = zext i32 %and to i64
  %shl = shl i64 %ext, 2
  store i64 %shl, i64 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sext_shl64_to_32:
; GCN: s_lshl_b32
; GCN-NOT: s_lshl_b64
define amdgpu_kernel void @sext_shl64_to_32(i64 addrspace(1)* nocapture %out, i32 %x) {
  %and = and i32 %x, 536870911
  %ext = sext i32 %and to i64
  %shl = shl i64 %ext, 2
  store i64 %shl, i64 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}zext_shl64_overflow:
; GCN: s_lshl_b64
; GCN-NOT: s_lshl_b32
define amdgpu_kernel void @zext_shl64_overflow(i64 addrspace(1)* nocapture %out, i32 %x) {
  %and = and i32 %x, 2147483647
  %ext = zext i32 %and to i64
  %shl = shl i64 %ext, 2
  store i64 %shl, i64 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sext_shl64_overflow:
; GCN: s_lshl_b64
; GCN-NOT: s_lshl_b32
define amdgpu_kernel void @sext_shl64_overflow(i64 addrspace(1)* nocapture %out, i32 %x) {
  %and = and i32 %x, 2147483647
  %ext = sext i32 %and to i64
  %shl = shl i64 %ext, 2
  store i64 %shl, i64 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}mulu24_shl64:
; GCN: v_mul_u32_u24_e32 [[M:v[0-9]+]], 7, v{{[0-9]+}}
; GCN: v_lshlrev_b32_e32 v{{[0-9]+}}, 2, [[M]]
define amdgpu_kernel void @mulu24_shl64(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = and i32 %tmp, 6
  %mulconv = mul nuw nsw i32 %tmp1, 7
  %tmp2 = zext i32 %mulconv to i64
  %tmp3 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp2
  store i32 0, i32 addrspace(1)* %tmp3, align 4
  ret void
}

; GCN-LABEL: {{^}}muli24_shl64:
; GCN: v_mul_i32_i24_e32 [[M:v[0-9]+]], -7, v{{[0-9]+}}
; GCN: v_lshlrev_b32_e32 v{{[0-9]+}}, 3, [[M]]
define amdgpu_kernel void @muli24_shl64(i64 addrspace(1)* nocapture %arg, i32 addrspace(1)* nocapture readonly %arg1) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp2
  %tmp4 = load i32, i32 addrspace(1)* %tmp3, align 4
  %tmp5 = or i32 %tmp4, -8388608
  %tmp6 = mul nsw i32 %tmp5, -7
  %tmp7 = zext i32 %tmp6 to i64
  %tmp8 = shl nuw nsw i64 %tmp7, 3
  %tmp9 = getelementptr inbounds i64, i64 addrspace(1)* %arg, i64 %tmp2
  store i64 %tmp8, i64 addrspace(1)* %tmp9, align 8
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
