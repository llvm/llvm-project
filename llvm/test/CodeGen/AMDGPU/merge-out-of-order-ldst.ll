; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

@L = external local_unnamed_addr addrspace(3) global [9 x double], align 16
@Ldisp = external local_unnamed_addr addrspace(3) global [96 x double], align 16

; Stores are reordered during loads merge. This case used to assert while
; scanning for a paired instruction because it used to expect paired one
; to follow a base one.

; GCN-LABEL: {{^}}out_of_order_merge:
; GCN-COUNT2: ds_read2_b64
; GCN-COUNT3: ds_write_b64
define amdgpu_kernel void @out_of_order_merge() {
entry:
  %gep1 = getelementptr inbounds [96 x double], [96 x double] addrspace(3)* @Ldisp, i32 0, i32 0
  %gep2 = getelementptr inbounds [96 x double], [96 x double] addrspace(3)* @Ldisp, i32 0, i32 1
  %tmp12 = load <2 x double>, <2 x double> addrspace(3)* bitcast (double addrspace(3)* getelementptr inbounds ([9 x double], [9 x double] addrspace(3)* @L, i32 0, i32 1) to <2 x double> addrspace(3)*), align 8
  %tmp14 = extractelement <2 x double> %tmp12, i32 0
  %tmp15 = extractelement <2 x double> %tmp12, i32 1
  %add50.i = fadd double %tmp14, %tmp15
  store double %add50.i, double addrspace(3)* %gep1, align 8
  %tmp16 = load double, double addrspace(3)* getelementptr inbounds ([9 x double], [9 x double] addrspace(3)* @L, i32 1, i32 0), align 8
  store double %tmp16, double addrspace(3)* %gep2, align 8
  %tmp17 = load <2 x double>, <2 x double> addrspace(3)* bitcast (double addrspace(3)* getelementptr inbounds ([9 x double], [9 x double] addrspace(3)* @L, i32 2, i32 1) to <2 x double> addrspace(3)*), align 8
  %tmp19 = extractelement <2 x double> %tmp17, i32 1
  store double %tmp19, double addrspace(3)* undef, align 8
  ret void
}
