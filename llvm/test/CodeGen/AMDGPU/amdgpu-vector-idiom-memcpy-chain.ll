; RUN: opt -amdgpu-vector-idiom-enable -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-vector-idiom -S %s | FileCheck %s
;
; Reduced testcase for planned enhancement:
; Fold a memcpy chain where the source memcpy is fed by a select-of-pointers
; and the second memcpy copies from the tmp alloca to the final destination.
; Expected: eliminate both memcpys by emitting a value-level select of two
; vector loads followed by a single store to the final destination.
;
; Enhancement implemented: memcpy chain folding now works

declare void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg)

; -----------------------------------------------------------------------------
; memcpy(tmp <- select(pa, pb)); memcpy(dst <- tmp)
; Expect: load <4 x i32> from pa/pb, select, store to dst. No memcpy.
;
define amdgpu_kernel void @memcpy_chain_src_select_elide_tmp(i1 %cond) {
; CHECK-LABEL: define amdgpu_kernel void @memcpy_chain_src_select_elide_tmp(
; CHECK-SAME: i1 [[COND:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[PA:%.*]] = alloca [4 x i32], align 16, addrspace(5)
; CHECK-NEXT:    [[PB:%.*]] = alloca [4 x i32], align 16, addrspace(5)
; CHECK:         [[DST:%.*]] = alloca [4 x i32], align 16, addrspace(5)
; CHECK:         [[SRC:%.*]] = select i1 [[COND]], ptr addrspace(5) [[PA]], ptr addrspace(5) [[PB]]
; CHECK:         [[LA:%.*]] = load <4 x i32>, ptr addrspace(5) [[PA]], align 16
; CHECK:         [[LB:%.*]] = load <4 x i32>, ptr addrspace(5) [[PB]], align 16
; CHECK:         [[SEL:%.*]] = select i1 [[COND]], <4 x i32> [[LA]], <4 x i32> [[LB]]
; CHECK:         store <4 x i32> [[SEL]], ptr addrspace(5) [[DST]], align 16
; CHECK-NOT:     call void @llvm.memcpy
; CHECK:         ret void
;
entry:
  %pa = alloca [4 x i32], align 16, addrspace(5)
  %pb = alloca [4 x i32], align 16, addrspace(5)
  %dst = alloca [4 x i32], align 16, addrspace(5)
  %tmp = alloca [4 x i32], align 16, addrspace(5)
  %src = select i1 %cond, ptr addrspace(5) %pa, ptr addrspace(5) %pb
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 16 %tmp, ptr addrspace(5) align 16 %src, i64 16, i1 false)
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 16 %dst, ptr addrspace(5) align 16 %tmp, i64 16, i1 false)
  ret void
}
