; RUN: opt -mtriple=amdgpu9.50-amd-amdhsa -passes=loop-unroll -S %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt -mtriple=amdgpu9.50-amd-amdhsa -passes=loop-unroll \
; RUN:   -amdgpu-unroll-runtime-local=false -S %s | FileCheck %s --check-prefixes=CHECK,NOLOCAL

; A convergent MFMA K-loop reading LDS (addrspace(3)) with a runtime trip
; count, modeled on a triton bf16 GEMM inner loop. By default it is runtime
; unrolled; with -amdgpu-unroll-runtime-local off the LDS loop is not.

@global_smem = external addrspace(3) global [0 x i8], align 16

; CHECK-LABEL: @gemm_k_loop(
; DEFAULT: loop.epil:
; NOLOCAL-NOT: loop.epil
define amdgpu_kernel void @gemm_k_loop(<8 x bfloat> %a, i32 %n) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi <4 x float> [ zeroinitializer, %entry ], [ %mfma, %loop ]
  %off = shl i32 %iv, 4
  %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %off
  %b = load <8 x bfloat>, ptr addrspace(3) %ptr, align 16
  %mfma = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %a, <8 x bfloat> %b, <4 x float> %acc, i32 0, i32 0, i32 0)
  %iv.next = add nuw nsw i32 %iv, 1
  %exit = icmp eq i32 %iv.next, %n
  br i1 %exit, label %end, label %loop

end:
  ret void
}

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)
