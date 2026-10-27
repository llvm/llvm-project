; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -O3 -global-isel=0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -O3 -global-isel=1 -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -O3 -global-isel=0 -amdgpu-disable-rewrite-mfma-form-sched-stage=false -verify-machineinstrs < %s | FileCheck %s
;
; Derived from the 4x4 accumulator reproducer in
; https://gist.github.com/bangtianliu/2ee1929472af01ce000bbebb912150a7
; (revision 5351ad516cba0b047f8b2738b9a9c951db2a9ce0).
; Replace the C/D boundary pins with require_agpr on every native MFMA.
; Sixteen tiles stay live across four K steps, and their results are stored.
; Check actual post-RA C/D operands, including immediate zero on the first step.
;
; CHECK-LABEL: mfma_agpr_pins:
; CHECK-COUNT-64: v_mfma_f32_16x16x32_f16 a[{{[0-9]+:[0-9]+}}], {{.*}}, {{(a\[[0-9]+:[0-9]+\]|0)}}{{$}}
; CHECK: ; NumAgprs: 64
; CHECK: ; ScratchSize: 0

target triple = "amdgcn-amd-amdhsa"

@lds_a_buf = internal addrspace(3) global [4096 x <8 x half>] poison, align 16
@lds_b_buf = internal addrspace(3) global [4096 x <8 x half>] poison, align 16

define amdgpu_kernel void @mfma_agpr_pins(ptr addrspace(1) %out, i32 %ksteps) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %lds_a = getelementptr <8 x half>, ptr addrspace(3) @lds_a_buf, i32 %tid
  %lds_b = getelementptr <8 x half>, ptr addrspace(3) @lds_b_buf, i32 %tid
  %t1 = add i32 0, 0
  %t2 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t1
  %s0_a0 = load <8 x half>, ptr addrspace(3) %t2, align 16
  %t3 = add i32 0, 64
  %t4 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t3
  %s0_a1 = load <8 x half>, ptr addrspace(3) %t4, align 16
  %t5 = add i32 0, 128
  %t6 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t5
  %s0_a2 = load <8 x half>, ptr addrspace(3) %t6, align 16
  %t7 = add i32 0, 192
  %t8 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t7
  %s0_a3 = load <8 x half>, ptr addrspace(3) %t8, align 16
  %t9 = add i32 0, 0
  %t10 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t9
  %s0_b0 = load <8 x half>, ptr addrspace(3) %t10, align 16
  %t11 = add i32 0, 64
  %t12 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t11
  %s0_b1 = load <8 x half>, ptr addrspace(3) %t12, align 16
  %t13 = add i32 0, 128
  %t14 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t13
  %s0_b2 = load <8 x half>, ptr addrspace(3) %t14, align 16
  %t15 = add i32 0, 192
  %t16 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t15
  %s0_b3 = load <8 x half>, ptr addrspace(3) %t16, align 16
  %t17 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a0, <8 x half> %s0_b0, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t18 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a0, <8 x half> %s0_b1, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t19 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a0, <8 x half> %s0_b2, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t20 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a0, <8 x half> %s0_b3, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t21 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a1, <8 x half> %s0_b0, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t22 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a1, <8 x half> %s0_b1, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t23 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a1, <8 x half> %s0_b2, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t24 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a1, <8 x half> %s0_b3, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t25 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a2, <8 x half> %s0_b0, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t26 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a2, <8 x half> %s0_b1, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t27 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a2, <8 x half> %s0_b2, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t28 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a2, <8 x half> %s0_b3, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t29 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a3, <8 x half> %s0_b0, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t30 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a3, <8 x half> %s0_b1, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t31 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a3, <8 x half> %s0_b2, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t32 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s0_a3, <8 x half> %s0_b3, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t33 = add i32 8, 0
  %t34 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t33
  %s1_a0 = load <8 x half>, ptr addrspace(3) %t34, align 16
  %t35 = add i32 8, 64
  %t36 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t35
  %s1_a1 = load <8 x half>, ptr addrspace(3) %t36, align 16
  %t37 = add i32 8, 128
  %t38 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t37
  %s1_a2 = load <8 x half>, ptr addrspace(3) %t38, align 16
  %t39 = add i32 8, 192
  %t40 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t39
  %s1_a3 = load <8 x half>, ptr addrspace(3) %t40, align 16
  %t41 = add i32 8, 0
  %t42 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t41
  %s1_b0 = load <8 x half>, ptr addrspace(3) %t42, align 16
  %t43 = add i32 8, 64
  %t44 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t43
  %s1_b1 = load <8 x half>, ptr addrspace(3) %t44, align 16
  %t45 = add i32 8, 128
  %t46 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t45
  %s1_b2 = load <8 x half>, ptr addrspace(3) %t46, align 16
  %t47 = add i32 8, 192
  %t48 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t47
  %s1_b3 = load <8 x half>, ptr addrspace(3) %t48, align 16
  %t49 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a0, <8 x half> %s1_b0, <4 x float> %t17, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t50 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a0, <8 x half> %s1_b1, <4 x float> %t18, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t51 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a0, <8 x half> %s1_b2, <4 x float> %t19, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t52 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a0, <8 x half> %s1_b3, <4 x float> %t20, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t53 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a1, <8 x half> %s1_b0, <4 x float> %t21, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t54 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a1, <8 x half> %s1_b1, <4 x float> %t22, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t55 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a1, <8 x half> %s1_b2, <4 x float> %t23, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t56 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a1, <8 x half> %s1_b3, <4 x float> %t24, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t57 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a2, <8 x half> %s1_b0, <4 x float> %t25, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t58 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a2, <8 x half> %s1_b1, <4 x float> %t26, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t59 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a2, <8 x half> %s1_b2, <4 x float> %t27, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t60 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a2, <8 x half> %s1_b3, <4 x float> %t28, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t61 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a3, <8 x half> %s1_b0, <4 x float> %t29, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t62 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a3, <8 x half> %s1_b1, <4 x float> %t30, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t63 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a3, <8 x half> %s1_b2, <4 x float> %t31, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t64 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s1_a3, <8 x half> %s1_b3, <4 x float> %t32, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t65 = add i32 16, 0
  %t66 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t65
  %s2_a0 = load <8 x half>, ptr addrspace(3) %t66, align 16
  %t67 = add i32 16, 64
  %t68 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t67
  %s2_a1 = load <8 x half>, ptr addrspace(3) %t68, align 16
  %t69 = add i32 16, 128
  %t70 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t69
  %s2_a2 = load <8 x half>, ptr addrspace(3) %t70, align 16
  %t71 = add i32 16, 192
  %t72 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t71
  %s2_a3 = load <8 x half>, ptr addrspace(3) %t72, align 16
  %t73 = add i32 16, 0
  %t74 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t73
  %s2_b0 = load <8 x half>, ptr addrspace(3) %t74, align 16
  %t75 = add i32 16, 64
  %t76 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t75
  %s2_b1 = load <8 x half>, ptr addrspace(3) %t76, align 16
  %t77 = add i32 16, 128
  %t78 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t77
  %s2_b2 = load <8 x half>, ptr addrspace(3) %t78, align 16
  %t79 = add i32 16, 192
  %t80 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t79
  %s2_b3 = load <8 x half>, ptr addrspace(3) %t80, align 16
  %t81 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a0, <8 x half> %s2_b0, <4 x float> %t49, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t82 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a0, <8 x half> %s2_b1, <4 x float> %t50, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t83 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a0, <8 x half> %s2_b2, <4 x float> %t51, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t84 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a0, <8 x half> %s2_b3, <4 x float> %t52, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t85 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a1, <8 x half> %s2_b0, <4 x float> %t53, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t86 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a1, <8 x half> %s2_b1, <4 x float> %t54, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t87 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a1, <8 x half> %s2_b2, <4 x float> %t55, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t88 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a1, <8 x half> %s2_b3, <4 x float> %t56, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t89 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a2, <8 x half> %s2_b0, <4 x float> %t57, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t90 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a2, <8 x half> %s2_b1, <4 x float> %t58, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t91 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a2, <8 x half> %s2_b2, <4 x float> %t59, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t92 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a2, <8 x half> %s2_b3, <4 x float> %t60, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t93 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a3, <8 x half> %s2_b0, <4 x float> %t61, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t94 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a3, <8 x half> %s2_b1, <4 x float> %t62, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t95 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a3, <8 x half> %s2_b2, <4 x float> %t63, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t96 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s2_a3, <8 x half> %s2_b3, <4 x float> %t64, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t97 = add i32 24, 0
  %t98 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t97
  %s3_a0 = load <8 x half>, ptr addrspace(3) %t98, align 16
  %t99 = add i32 24, 64
  %t100 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t99
  %s3_a1 = load <8 x half>, ptr addrspace(3) %t100, align 16
  %t101 = add i32 24, 128
  %t102 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t101
  %s3_a2 = load <8 x half>, ptr addrspace(3) %t102, align 16
  %t103 = add i32 24, 192
  %t104 = getelementptr <8 x half>, ptr addrspace(3) %lds_a, i32 %t103
  %s3_a3 = load <8 x half>, ptr addrspace(3) %t104, align 16
  %t105 = add i32 24, 0
  %t106 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t105
  %s3_b0 = load <8 x half>, ptr addrspace(3) %t106, align 16
  %t107 = add i32 24, 64
  %t108 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t107
  %s3_b1 = load <8 x half>, ptr addrspace(3) %t108, align 16
  %t109 = add i32 24, 128
  %t110 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t109
  %s3_b2 = load <8 x half>, ptr addrspace(3) %t110, align 16
  %t111 = add i32 24, 192
  %t112 = getelementptr <8 x half>, ptr addrspace(3) %lds_b, i32 %t111
  %s3_b3 = load <8 x half>, ptr addrspace(3) %t112, align 16
  %t113 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a0, <8 x half> %s3_b0, <4 x float> %t81, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t114 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a0, <8 x half> %s3_b1, <4 x float> %t82, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t115 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a0, <8 x half> %s3_b2, <4 x float> %t83, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t116 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a0, <8 x half> %s3_b3, <4 x float> %t84, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t117 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a1, <8 x half> %s3_b0, <4 x float> %t85, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t118 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a1, <8 x half> %s3_b1, <4 x float> %t86, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t119 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a1, <8 x half> %s3_b2, <4 x float> %t87, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t120 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a1, <8 x half> %s3_b3, <4 x float> %t88, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t121 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a2, <8 x half> %s3_b0, <4 x float> %t89, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t122 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a2, <8 x half> %s3_b1, <4 x float> %t90, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t123 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a2, <8 x half> %s3_b2, <4 x float> %t91, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t124 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a2, <8 x half> %s3_b3, <4 x float> %t92, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t125 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a3, <8 x half> %s3_b0, <4 x float> %t93, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t126 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a3, <8 x half> %s3_b1, <4 x float> %t94, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t127 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a3, <8 x half> %s3_b2, <4 x float> %t95, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %t128 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %s3_a3, <8 x half> %s3_b3, <4 x float> %t96, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %o0 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 0
  %o0.lane = getelementptr <4 x float>, ptr addrspace(1) %o0, i32 %tid
  store <4 x float> %t113, ptr addrspace(1) %o0.lane, align 16
  %o1 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 256
  %o1.lane = getelementptr <4 x float>, ptr addrspace(1) %o1, i32 %tid
  store <4 x float> %t114, ptr addrspace(1) %o1.lane, align 16
  %o2 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 512
  %o2.lane = getelementptr <4 x float>, ptr addrspace(1) %o2, i32 %tid
  store <4 x float> %t115, ptr addrspace(1) %o2.lane, align 16
  %o3 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 768
  %o3.lane = getelementptr <4 x float>, ptr addrspace(1) %o3, i32 %tid
  store <4 x float> %t116, ptr addrspace(1) %o3.lane, align 16
  %o4 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 1024
  %o4.lane = getelementptr <4 x float>, ptr addrspace(1) %o4, i32 %tid
  store <4 x float> %t117, ptr addrspace(1) %o4.lane, align 16
  %o5 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 1280
  %o5.lane = getelementptr <4 x float>, ptr addrspace(1) %o5, i32 %tid
  store <4 x float> %t118, ptr addrspace(1) %o5.lane, align 16
  %o6 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 1536
  %o6.lane = getelementptr <4 x float>, ptr addrspace(1) %o6, i32 %tid
  store <4 x float> %t119, ptr addrspace(1) %o6.lane, align 16
  %o7 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 1792
  %o7.lane = getelementptr <4 x float>, ptr addrspace(1) %o7, i32 %tid
  store <4 x float> %t120, ptr addrspace(1) %o7.lane, align 16
  %o8 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 2048
  %o8.lane = getelementptr <4 x float>, ptr addrspace(1) %o8, i32 %tid
  store <4 x float> %t121, ptr addrspace(1) %o8.lane, align 16
  %o9 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 2304
  %o9.lane = getelementptr <4 x float>, ptr addrspace(1) %o9, i32 %tid
  store <4 x float> %t122, ptr addrspace(1) %o9.lane, align 16
  %o10 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 2560
  %o10.lane = getelementptr <4 x float>, ptr addrspace(1) %o10, i32 %tid
  store <4 x float> %t123, ptr addrspace(1) %o10.lane, align 16
  %o11 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 2816
  %o11.lane = getelementptr <4 x float>, ptr addrspace(1) %o11, i32 %tid
  store <4 x float> %t124, ptr addrspace(1) %o11.lane, align 16
  %o12 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 3072
  %o12.lane = getelementptr <4 x float>, ptr addrspace(1) %o12, i32 %tid
  store <4 x float> %t125, ptr addrspace(1) %o12.lane, align 16
  %o13 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 3328
  %o13.lane = getelementptr <4 x float>, ptr addrspace(1) %o13, i32 %tid
  store <4 x float> %t126, ptr addrspace(1) %o13.lane, align 16
  %o14 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 3584
  %o14.lane = getelementptr <4 x float>, ptr addrspace(1) %o14, i32 %tid
  store <4 x float> %t127, ptr addrspace(1) %o14.lane, align 16
  %o15 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 3840
  %o15.lane = getelementptr <4 x float>, ptr addrspace(1) %o15, i32 %tid
  store <4 x float> %t128, ptr addrspace(1) %o15.lane, align 16
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32, i32, i32)

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1,1" }
