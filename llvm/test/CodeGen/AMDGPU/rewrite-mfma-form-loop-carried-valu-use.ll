; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -verify-machineinstrs -amdgpu-rewrite-mfma-form-sched-stage=true < %s | FileCheck %s

; A loop-carried accumulator converted to f16 after the loop, which shares one
; register with the zero-init and the MFMAs after coalescing. The conversion
; only needs an AGPR-to-VGPR copy after the loop, so the MFMAs use AGPR form.

; CHECK-LABEL: loop_carried_acc_valu_use:
; CHECK: [[LOOP:\.LBB0_[0-9]+]]: ; %loop
; CHECK-NOT: v_accvgpr
; CHECK: v_mfma_f32_32x32x16_f16 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}]
; CHECK-NOT: v_accvgpr
; CHECK: s_cbranch_scc1 [[LOOP]]
; CHECK: v_accvgpr_read_b32
; CHECK: v_cvt_pk_f16_f32

define amdgpu_kernel void @loop_carried_acc_valu_use(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %n) #0 {
entry:
  %skip = icmp slt i32 %n, 1
  br i1 %skip, label %exit, label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc0 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc0.next, %loop ]
  %acc1 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc1.next, %loop ]
  %acc2 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc2.next, %loop ]
  %acc3 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc3.next, %loop ]
  %acc4 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc4.next, %loop ]
  %acc5 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc5.next, %loop ]
  %acc6 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc6.next, %loop ]
  %acc7 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc7.next, %loop ]
  %acc8 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc8.next, %loop ]
  %acc9 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc9.next, %loop ]
  %acc10 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc10.next, %loop ]
  %acc11 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc11.next, %loop ]
  %acc12 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc12.next, %loop ]
  %acc13 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc13.next, %loop ]
  %acc14 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc14.next, %loop ]
  %acc15 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc15.next, %loop ]
  %a.ptr0 = getelementptr <8 x half>, ptr addrspace(1) %a, i32 %i
  %a.ptr1 = getelementptr <8 x half>, ptr addrspace(1) %a.ptr0, i32 1
  %b.ptr0 = getelementptr <8 x half>, ptr addrspace(1) %b, i32 %i
  %b.ptr1 = getelementptr <8 x half>, ptr addrspace(1) %b.ptr0, i32 1
  %a0 = load <8 x half>, ptr addrspace(1) %a.ptr0
  %b0 = load <8 x half>, ptr addrspace(1) %b.ptr0
  %a1 = load <8 x half>, ptr addrspace(1) %a.ptr1
  %b1 = load <8 x half>, ptr addrspace(1) %b.ptr1
  %mid0 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc0, i32 0, i32 0, i32 0)
  %acc0.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid0, i32 0, i32 0, i32 0)
  %mid1 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc1, i32 0, i32 0, i32 0)
  %acc1.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid1, i32 0, i32 0, i32 0)
  %mid2 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc2, i32 0, i32 0, i32 0)
  %acc2.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid2, i32 0, i32 0, i32 0)
  %mid3 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc3, i32 0, i32 0, i32 0)
  %acc3.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid3, i32 0, i32 0, i32 0)
  %mid4 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc4, i32 0, i32 0, i32 0)
  %acc4.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid4, i32 0, i32 0, i32 0)
  %mid5 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc5, i32 0, i32 0, i32 0)
  %acc5.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid5, i32 0, i32 0, i32 0)
  %mid6 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc6, i32 0, i32 0, i32 0)
  %acc6.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid6, i32 0, i32 0, i32 0)
  %mid7 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc7, i32 0, i32 0, i32 0)
  %acc7.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid7, i32 0, i32 0, i32 0)
  %mid8 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc8, i32 0, i32 0, i32 0)
  %acc8.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid8, i32 0, i32 0, i32 0)
  %mid9 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc9, i32 0, i32 0, i32 0)
  %acc9.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid9, i32 0, i32 0, i32 0)
  %mid10 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc10, i32 0, i32 0, i32 0)
  %acc10.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid10, i32 0, i32 0, i32 0)
  %mid11 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc11, i32 0, i32 0, i32 0)
  %acc11.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid11, i32 0, i32 0, i32 0)
  %mid12 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc12, i32 0, i32 0, i32 0)
  %acc12.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid12, i32 0, i32 0, i32 0)
  %mid13 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc13, i32 0, i32 0, i32 0)
  %acc13.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid13, i32 0, i32 0, i32 0)
  %mid14 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc14, i32 0, i32 0, i32 0)
  %acc14.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid14, i32 0, i32 0, i32 0)
  %mid15 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc15, i32 0, i32 0, i32 0)
  %acc15.next = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid15, i32 0, i32 0, i32 0)
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit, !prof !0

exit:
  %res0 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc0.next, %loop ]
  %res1 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc1.next, %loop ]
  %res2 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc2.next, %loop ]
  %res3 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc3.next, %loop ]
  %res4 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc4.next, %loop ]
  %res5 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc5.next, %loop ]
  %res6 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc6.next, %loop ]
  %res7 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc7.next, %loop ]
  %res8 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc8.next, %loop ]
  %res9 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc9.next, %loop ]
  %res10 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc10.next, %loop ]
  %res11 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc11.next, %loop ]
  %res12 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc12.next, %loop ]
  %res13 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc13.next, %loop ]
  %res14 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc14.next, %loop ]
  %res15 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc15.next, %loop ]
  %half0 = fptrunc <16 x float> %res0 to <16 x half>
  %out.ptr0 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 0
  store <16 x half> %half0, ptr addrspace(1) %out.ptr0
  %half1 = fptrunc <16 x float> %res1 to <16 x half>
  %out.ptr1 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 1
  store <16 x half> %half1, ptr addrspace(1) %out.ptr1
  %half2 = fptrunc <16 x float> %res2 to <16 x half>
  %out.ptr2 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 2
  store <16 x half> %half2, ptr addrspace(1) %out.ptr2
  %half3 = fptrunc <16 x float> %res3 to <16 x half>
  %out.ptr3 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 3
  store <16 x half> %half3, ptr addrspace(1) %out.ptr3
  %half4 = fptrunc <16 x float> %res4 to <16 x half>
  %out.ptr4 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 4
  store <16 x half> %half4, ptr addrspace(1) %out.ptr4
  %half5 = fptrunc <16 x float> %res5 to <16 x half>
  %out.ptr5 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 5
  store <16 x half> %half5, ptr addrspace(1) %out.ptr5
  %half6 = fptrunc <16 x float> %res6 to <16 x half>
  %out.ptr6 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 6
  store <16 x half> %half6, ptr addrspace(1) %out.ptr6
  %half7 = fptrunc <16 x float> %res7 to <16 x half>
  %out.ptr7 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 7
  store <16 x half> %half7, ptr addrspace(1) %out.ptr7
  %half8 = fptrunc <16 x float> %res8 to <16 x half>
  %out.ptr8 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 8
  store <16 x half> %half8, ptr addrspace(1) %out.ptr8
  %half9 = fptrunc <16 x float> %res9 to <16 x half>
  %out.ptr9 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 9
  store <16 x half> %half9, ptr addrspace(1) %out.ptr9
  %half10 = fptrunc <16 x float> %res10 to <16 x half>
  %out.ptr10 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 10
  store <16 x half> %half10, ptr addrspace(1) %out.ptr10
  %half11 = fptrunc <16 x float> %res11 to <16 x half>
  %out.ptr11 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 11
  store <16 x half> %half11, ptr addrspace(1) %out.ptr11
  %half12 = fptrunc <16 x float> %res12 to <16 x half>
  %out.ptr12 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 12
  store <16 x half> %half12, ptr addrspace(1) %out.ptr12
  %half13 = fptrunc <16 x float> %res13 to <16 x half>
  %out.ptr13 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 13
  store <16 x half> %half13, ptr addrspace(1) %out.ptr13
  %half14 = fptrunc <16 x float> %res14 to <16 x half>
  %out.ptr14 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 14
  store <16 x half> %half14, ptr addrspace(1) %out.ptr14
  %half15 = fptrunc <16 x float> %res15 to <16 x half>
  %out.ptr15 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 15
  store <16 x half> %half15, ptr addrspace(1) %out.ptr15
  ret void
}

; The accumulator is rescaled inside the loop, so a copy back to VGPRs would
; run every iteration. The MFMAs whose result is rescaled stay in VGPR form.

; CHECK-LABEL: loop_carried_acc_rescaled:
; CHECK: [[LOOP:\.LBB1_[0-9]+]]: ; %loop
; CHECK-NOT: v_mfma_f32_32x32x16_f16 a[
; CHECK: v_mfma_f32_32x32x16_f16 v[{{[0-9]+:[0-9]+}}], {{[av]}}[{{[0-9]+:[0-9]+}}], {{[av]}}[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; CHECK: s_cbranch_scc1 [[LOOP]]

define amdgpu_kernel void @loop_carried_acc_rescaled(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %n, <16 x float> %scale) #0 {
entry:
  %skip = icmp slt i32 %n, 1
  br i1 %skip, label %exit, label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc0 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc0.next, %loop ]
  %acc1 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc1.next, %loop ]
  %acc2 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc2.next, %loop ]
  %acc3 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc3.next, %loop ]
  %acc4 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc4.next, %loop ]
  %acc5 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc5.next, %loop ]
  %acc6 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc6.next, %loop ]
  %acc7 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc7.next, %loop ]
  %acc8 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc8.next, %loop ]
  %acc9 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc9.next, %loop ]
  %acc10 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc10.next, %loop ]
  %acc11 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc11.next, %loop ]
  %acc12 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc12.next, %loop ]
  %acc13 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc13.next, %loop ]
  %acc14 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc14.next, %loop ]
  %acc15 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc15.next, %loop ]
  %a.ptr0 = getelementptr <8 x half>, ptr addrspace(1) %a, i32 %i
  %a.ptr1 = getelementptr <8 x half>, ptr addrspace(1) %a.ptr0, i32 1
  %b.ptr0 = getelementptr <8 x half>, ptr addrspace(1) %b, i32 %i
  %b.ptr1 = getelementptr <8 x half>, ptr addrspace(1) %b.ptr0, i32 1
  %a0 = load <8 x half>, ptr addrspace(1) %a.ptr0
  %b0 = load <8 x half>, ptr addrspace(1) %b.ptr0
  %a1 = load <8 x half>, ptr addrspace(1) %a.ptr1
  %b1 = load <8 x half>, ptr addrspace(1) %b.ptr1
  %mid0 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc0, i32 0, i32 0, i32 0)
  %prod0 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid0, i32 0, i32 0, i32 0)
  %acc0.next = fmul <16 x float> %prod0, %scale
  %mid1 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc1, i32 0, i32 0, i32 0)
  %prod1 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid1, i32 0, i32 0, i32 0)
  %acc1.next = fmul <16 x float> %prod1, %scale
  %mid2 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc2, i32 0, i32 0, i32 0)
  %prod2 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid2, i32 0, i32 0, i32 0)
  %acc2.next = fmul <16 x float> %prod2, %scale
  %mid3 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc3, i32 0, i32 0, i32 0)
  %prod3 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid3, i32 0, i32 0, i32 0)
  %acc3.next = fmul <16 x float> %prod3, %scale
  %mid4 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc4, i32 0, i32 0, i32 0)
  %prod4 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid4, i32 0, i32 0, i32 0)
  %acc4.next = fmul <16 x float> %prod4, %scale
  %mid5 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc5, i32 0, i32 0, i32 0)
  %prod5 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid5, i32 0, i32 0, i32 0)
  %acc5.next = fmul <16 x float> %prod5, %scale
  %mid6 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc6, i32 0, i32 0, i32 0)
  %prod6 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid6, i32 0, i32 0, i32 0)
  %acc6.next = fmul <16 x float> %prod6, %scale
  %mid7 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc7, i32 0, i32 0, i32 0)
  %prod7 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid7, i32 0, i32 0, i32 0)
  %acc7.next = fmul <16 x float> %prod7, %scale
  %mid8 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc8, i32 0, i32 0, i32 0)
  %prod8 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid8, i32 0, i32 0, i32 0)
  %acc8.next = fmul <16 x float> %prod8, %scale
  %mid9 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc9, i32 0, i32 0, i32 0)
  %prod9 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid9, i32 0, i32 0, i32 0)
  %acc9.next = fmul <16 x float> %prod9, %scale
  %mid10 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc10, i32 0, i32 0, i32 0)
  %prod10 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid10, i32 0, i32 0, i32 0)
  %acc10.next = fmul <16 x float> %prod10, %scale
  %mid11 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc11, i32 0, i32 0, i32 0)
  %prod11 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid11, i32 0, i32 0, i32 0)
  %acc11.next = fmul <16 x float> %prod11, %scale
  %mid12 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc12, i32 0, i32 0, i32 0)
  %prod12 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid12, i32 0, i32 0, i32 0)
  %acc12.next = fmul <16 x float> %prod12, %scale
  %mid13 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc13, i32 0, i32 0, i32 0)
  %prod13 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid13, i32 0, i32 0, i32 0)
  %acc13.next = fmul <16 x float> %prod13, %scale
  %mid14 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc14, i32 0, i32 0, i32 0)
  %prod14 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid14, i32 0, i32 0, i32 0)
  %acc14.next = fmul <16 x float> %prod14, %scale
  %mid15 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a0, <8 x half> %b0, <16 x float> %acc15, i32 0, i32 0, i32 0)
  %prod15 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a1, <8 x half> %b1, <16 x float> %mid15, i32 0, i32 0, i32 0)
  %acc15.next = fmul <16 x float> %prod15, %scale
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit, !prof !0

exit:
  %res0 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc0.next, %loop ]
  %res1 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc1.next, %loop ]
  %res2 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc2.next, %loop ]
  %res3 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc3.next, %loop ]
  %res4 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc4.next, %loop ]
  %res5 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc5.next, %loop ]
  %res6 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc6.next, %loop ]
  %res7 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc7.next, %loop ]
  %res8 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc8.next, %loop ]
  %res9 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc9.next, %loop ]
  %res10 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc10.next, %loop ]
  %res11 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc11.next, %loop ]
  %res12 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc12.next, %loop ]
  %res13 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc13.next, %loop ]
  %res14 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc14.next, %loop ]
  %res15 = phi <16 x float> [ zeroinitializer, %entry ], [ %acc15.next, %loop ]
  %half0 = fptrunc <16 x float> %res0 to <16 x half>
  %out.ptr0 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 0
  store <16 x half> %half0, ptr addrspace(1) %out.ptr0
  %half1 = fptrunc <16 x float> %res1 to <16 x half>
  %out.ptr1 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 1
  store <16 x half> %half1, ptr addrspace(1) %out.ptr1
  %half2 = fptrunc <16 x float> %res2 to <16 x half>
  %out.ptr2 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 2
  store <16 x half> %half2, ptr addrspace(1) %out.ptr2
  %half3 = fptrunc <16 x float> %res3 to <16 x half>
  %out.ptr3 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 3
  store <16 x half> %half3, ptr addrspace(1) %out.ptr3
  %half4 = fptrunc <16 x float> %res4 to <16 x half>
  %out.ptr4 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 4
  store <16 x half> %half4, ptr addrspace(1) %out.ptr4
  %half5 = fptrunc <16 x float> %res5 to <16 x half>
  %out.ptr5 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 5
  store <16 x half> %half5, ptr addrspace(1) %out.ptr5
  %half6 = fptrunc <16 x float> %res6 to <16 x half>
  %out.ptr6 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 6
  store <16 x half> %half6, ptr addrspace(1) %out.ptr6
  %half7 = fptrunc <16 x float> %res7 to <16 x half>
  %out.ptr7 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 7
  store <16 x half> %half7, ptr addrspace(1) %out.ptr7
  %half8 = fptrunc <16 x float> %res8 to <16 x half>
  %out.ptr8 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 8
  store <16 x half> %half8, ptr addrspace(1) %out.ptr8
  %half9 = fptrunc <16 x float> %res9 to <16 x half>
  %out.ptr9 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 9
  store <16 x half> %half9, ptr addrspace(1) %out.ptr9
  %half10 = fptrunc <16 x float> %res10 to <16 x half>
  %out.ptr10 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 10
  store <16 x half> %half10, ptr addrspace(1) %out.ptr10
  %half11 = fptrunc <16 x float> %res11 to <16 x half>
  %out.ptr11 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 11
  store <16 x half> %half11, ptr addrspace(1) %out.ptr11
  %half12 = fptrunc <16 x float> %res12 to <16 x half>
  %out.ptr12 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 12
  store <16 x half> %half12, ptr addrspace(1) %out.ptr12
  %half13 = fptrunc <16 x float> %res13 to <16 x half>
  %out.ptr13 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 13
  store <16 x half> %half13, ptr addrspace(1) %out.ptr13
  %half14 = fptrunc <16 x float> %res14 to <16 x half>
  %out.ptr14 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 14
  store <16 x half> %half14, ptr addrspace(1) %out.ptr14
  %half15 = fptrunc <16 x float> %res15 to <16 x half>
  %out.ptr15 = getelementptr <16 x half>, ptr addrspace(1) %out, i32 15
  store <16 x half> %half15, ptr addrspace(1) %out.ptr15
  ret void
}

declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half>, <8 x half>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)

attributes #0 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" }

!0 = !{!"branch_weights", i32 1000, i32 1}
