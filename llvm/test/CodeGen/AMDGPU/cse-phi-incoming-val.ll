; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs | FileCheck %s

; Check that the redundant immediate MOV instruction
; (by-product of handling phi nodes) is not found
; in the generated code.

; CHECK-LABEL: {{^}}mov_opt:
; CHECK: s_mov_b32 [[SREG:s[0-9]+]], 1.0
; CHECK: %bb.1:
; CHECK-NOT: v_mov_b32_e32 {{v[0-9]+}}, 1.0
; CHECK: BB0_4:
; CHECK: v_mov_b32_e32 v{{[0-9]+}}, [[SREG]]

define amdgpu_ps void @mov_opt(i32 %arg, i32 inreg %arg1, i32 inreg %arg2) local_unnamed_addr #0 {
bb:
  %tmp = icmp eq i32 %arg1, 0
  br i1 %tmp, label %bb3, label %bb10

bb3:                                              ; preds = %bb
  %tmp4 = icmp eq i32 %arg2, 0
  br i1 %tmp4, label %bb5, label %bb10

bb5:                                              ; preds = %bb3
  %tmp6 = getelementptr <{ [4294967295 x i32] }>, <{ [4294967295 x i32] }> addrspace(6)* null, i32 0, i32 0, i32 %arg
  %tmp7 = load i32, i32 addrspace(6)* %tmp6
  %tmp8 = icmp eq i32 %tmp7, 1
  br i1 %tmp8, label %bb10, label %bb9

bb9:                                              ; preds = %bb5
  br label %bb10

bb10:                                             ; preds = %bb9, %bb5, %bb3, %bb
  %tmp11 = phi float [ 1.000000e+00, %bb3 ], [ 0.000000e+00, %bb9 ], [ 1.000000e+00, %bb ], [ undef, %bb5 ]
  call void @llvm.amdgcn.exp.f32(i32 immarg 40, i32 immarg 15, float %tmp11, float undef, float undef, float undef, i1 immarg false, i1 immarg false) #0
  ret void
}

; Function Attrs: inaccessiblememonly nounwind
declare void @llvm.amdgcn.exp.f32(i32 immarg, i32 immarg, float, float, float, float, i1 immarg, i1 immarg) #1

attributes #0 = { nounwind }
attributes #1 = { inaccessiblememonly nounwind }
