; RUN: llc -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; If the workgroup id range is restricted, we should be able to use
; mad24 for the usual indexing pattern.

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.x() #0
declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0

; GCN-LABEL: {{^}}get_global_id_0:
; GCN: s_and_b32 [[WGSIZEX:s[0-9]+]], {{s[0-9]+}}, 0xffff
; GCN: s_mul_i32 [[MUL:s[0-9]+]], s10, [[WGSIZEX]]
; GCN: v_add_i32_e32 v{{[0-9]+}}, vcc, [[MUL]], v0
define amdgpu_kernel void @get_global_id_0(ptr addrspace(1) %out) #1 {
  %dispatch.ptr = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep = getelementptr inbounds i32, ptr addrspace(4) %dispatch.ptr, i64 1
  %workgroup.size.xy = load i32, ptr addrspace(4) %gep, align 4, !invariant.load !0
  %workgroup.size.x = and i32 %workgroup.size.xy, 65535

  %workitem.id.x = call i32 @llvm.amdgcn.workitem.id.x(), !range !1
  %workgroup.id.x = call i32 @llvm.amdgcn.workgroup.id.x(), !range !2

  %mul = mul i32 %workgroup.id.x, %workgroup.size.x
  %add = add i32 %mul, %workitem.id.x

  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

!0 = !{}
!1 = !{i32 0, i32 1024}
!2 = !{i32 0, i32 16777216}
