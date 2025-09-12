; RUN: opt -amdgpu-vector-idiom-enable -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-vector-idiom -S %s | FileCheck %s

; This test verifies the AMDGPUVectorIdiomCombinePass transforms:
; 1) memcpy with select-fed source into a value-level select between two loads,
;    followed by one store (when it's safe to speculate both loads).
; 2) memcpy with select-fed destination into a control-flow split with two memcpys.

@G0 = addrspace(1) global [4 x i32] zeroinitializer, align 16
@G1 = addrspace(1) global [4 x i32] zeroinitializer, align 16

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg)

; -----------------------------------------------------------------------------
; Source is a select. Expect value-level select of two <4 x i32> loads
; and a single store, with no remaining memcpy.
;
; CHECK-LABEL: @value_select_src(
; CHECK-NOT: call void @llvm.memcpy
; CHECK:      [[LA:%.+]] = load <4 x i32>, ptr addrspace(1) [[A:%.+]], align 16
; CHECK:      [[LB:%.+]] = load <4 x i32>, ptr addrspace(1) [[B:%.+]], align 16
; CHECK:      [[SEL:%.+]] = select i1 [[COND:%.+]], <4 x i32> [[LA]], <4 x i32> [[LB]]
; CHECK:      store <4 x i32> [[SEL]], ptr addrspace(1) [[DST:%.+]], align 16
define amdgpu_kernel void @value_select_src(ptr addrspace(1) %dst, i1 %cond) {
entry:
  ; Pointers to two 16-byte aligned buffers in the same addrspace(1).
  %pa = getelementptr inbounds [4 x i32], ptr addrspace(1) @G0, i64 0, i64 0
  %pb = getelementptr inbounds [4 x i32], ptr addrspace(1) @G1, i64 0, i64 0
  %src = select i1 %cond, ptr addrspace(1) %pa, ptr addrspace(1) %pb

  ; Provide explicit operand alignments so the pass can emit an aligned store.
  call void @llvm.memcpy.p1.p1.i64(
    ptr addrspace(1) align 16 %dst,
    ptr addrspace(1) align 16 %src,
    i64 16, i1 false)

  ret void
}

; -----------------------------------------------------------------------------
; Destination is a select. Expect CFG split with two memcpys guarded
; by a branch (we do not speculate stores in this pass).
;
; CHECK-LABEL: @dest_select_cfg_split(
; CHECK:       br i1 %cond, label %memcpy.then, label %memcpy.else
; CHECK:     memcpy.join:
; CHECK:       ret void
; CHECK:     memcpy.then:
; CHECK:       call void @llvm.memcpy.p1.p1.i64(
; CHECK:       br label %memcpy.join
; CHECK:     memcpy.else:
; CHECK:       call void @llvm.memcpy.p1.p1.i64(
; CHECK:       br label %memcpy.join
define amdgpu_kernel void @dest_select_cfg_split(ptr addrspace(1) %da, ptr addrspace(1) %db,
                                                 ptr addrspace(1) %src, i1 %cond) {
entry:
  %dst = select i1 %cond, ptr addrspace(1) %da, ptr addrspace(1) %db
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dst, ptr addrspace(1) %src, i64 16, i1 false)
  ret void
}

; -----------------------------------------------------------------------------
; Source is a select, 4 x double (32 bytes).
; Expect value-level select of two <4 x i64> loads and a single store, no memcpy.
;
; CHECK-LABEL: @value_select_src_4xd(
; CHECK-NOT: call void @llvm.memcpy
; CHECK:      [[LA4D:%.+]] = load <4 x i64>, ptr addrspace(1) {{%.+}}, align 32
; CHECK:      [[LB4D:%.+]] = load <4 x i64>, ptr addrspace(1) {{%.+}}, align 32
; CHECK:      [[SEL4D:%.+]] = select i1 {{%.+}}, <4 x i64> [[LA4D]], <4 x i64> [[LB4D]]
; CHECK:      store <4 x i64> [[SEL4D]], ptr addrspace(1) {{%.+}}, align 32
@G2 = addrspace(1) global [4 x double] zeroinitializer, align 32
@G3 = addrspace(1) global [4 x double] zeroinitializer, align 32
define amdgpu_kernel void @value_select_src_4xd(ptr addrspace(1) %dst, i1 %cond) {
entry:
  %pa = getelementptr inbounds [4 x double], ptr addrspace(1) @G2, i64 0, i64 0
  %pb = getelementptr inbounds [4 x double], ptr addrspace(1) @G3, i64 0, i64 0
  %src = select i1 %cond, ptr addrspace(1) %pa, ptr addrspace(1) %pb

  call void @llvm.memcpy.p1.p1.i64(
    ptr addrspace(1) align 32 %dst,
    ptr addrspace(1) align 32 %src,
    i64 32, i1 false)

  ret void
}

; -----------------------------------------------------------------------------
; Source is a select, 3 x char (3 bytes).
; Expect value-level select using <3 x i8> loads/stores, no memcpy.
;
; CHECK-LABEL: @value_select_src_3xc(
; CHECK-NOT: call void @llvm.memcpy
; CHECK:      [[LA3C:%.+]] = load <3 x i8>, ptr addrspace(1) {{%.+}}, align 1
; CHECK:      [[LB3C:%.+]] = load <3 x i8>, ptr addrspace(1) {{%.+}}, align 1
; CHECK:      [[SEL3C:%.+]] = select i1 {{%.+}}, <3 x i8> [[LA3C]], <3 x i8> [[LB3C]]
; CHECK:      store <3 x i8> [[SEL3C]], ptr addrspace(1) {{%.+}}, align 1
@G4 = addrspace(1) global [3 x i8] zeroinitializer, align 1
@G5 = addrspace(1) global [3 x i8] zeroinitializer, align 1
define amdgpu_kernel void @value_select_src_3xc(ptr addrspace(1) %dst, i1 %cond) {
entry:
  %pa = getelementptr inbounds [3 x i8], ptr addrspace(1) @G4, i64 0, i64 0
  %pb = getelementptr inbounds [3 x i8], ptr addrspace(1) @G5, i64 0, i64 0
  %src = select i1 %cond, ptr addrspace(1) %pa, ptr addrspace(1) %pb

  call void @llvm.memcpy.p1.p1.i64(
    ptr addrspace(1) align 1 %dst,
    ptr addrspace(1) align 1 %src,
    i64 3, i1 false)

  ret void
}
