; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs --stress-regalloc=10 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs --stress-regalloc=10 < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_remat_sgpr:
; GCN-NOT:     v_writelane_b32
; GCN:         {{^}}[[LOOP:BB[0-9_]+]]:
; GCN-COUNT-6: s_mov_b32 s{{[0-9]+}}, 0x
; GCN-NOT:     v_writelane_b32
; GCN:         s_cbranch_{{[^ ]+}} [[LOOP]]
; GCN: .sgpr_spill_count: 0
define amdgpu_kernel void @test_remat_sgpr(double addrspace(1)* %arg, double addrspace(1)* %arg1) {
bb:
  %i = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %i4 = phi i32 [ 0, %bb ], [ %i22, %bb3 ]
  %i5 = add nuw nsw i32 %i4, %i
  %i6 = zext i32 %i5 to i64
  %i7 = getelementptr inbounds double, double addrspace(1)* %arg, i64 %i6
  %i8 = load double, double addrspace(1)* %i7, align 8
  %i9 = fadd double %i8, 0x3EFC01997CC9E6B0
  %i10 = tail call double @llvm.fma.f64(double %i8, double %i9, double 0x3FBE25E43ABE935A)
  %i11 = tail call double @llvm.fma.f64(double %i10, double %i9, double 0x3FC110EF47E6C9C2)
  %i12 = tail call double @llvm.fma.f64(double %i11, double %i9, double 0x3FC3B13BCFA74449)
  %i13 = tail call double @llvm.fma.f64(double %i12, double %i9, double 0x3FC745D171BF3C30)
  %i14 = tail call double @llvm.fma.f64(double %i13, double %i9, double 0x3FCC71C71C7792CE)
  %i15 = tail call double @llvm.fma.f64(double %i14, double %i9, double 0x3FD24924924920DA)
  %i16 = tail call double @llvm.fma.f64(double %i15, double %i9, double 0x3FD999999999999C)
  %i17 = tail call double @llvm.fma.f64(double %i16, double %i9, double 0x3FD899999999899C)
  %i18 = tail call double @llvm.fma.f64(double %i17, double %i9, double 0x3FD799999999799C)
  %i19 = tail call double @llvm.fma.f64(double %i18, double %i9, double 0x3FD699999999699C)
  %i20 = tail call double @llvm.fma.f64(double %i19, double %i9, double 0x3FD599999999599C)
  %i21 = getelementptr inbounds double, double addrspace(1)* %arg1, i64 %i6
  store double %i19, double addrspace(1)* %i21, align 8
  %i22 = add nuw nsw i32 %i4, 1
  %i23 = icmp eq i32 %i22, 1024
  br i1 %i23, label %bb2, label %bb3
}

declare double @llvm.fma.f64(double, double, double)
declare i32 @llvm.amdgcn.workitem.id.x()
