; NOTE: Test cases for TDM descriptor optimization pass.
;
; The pass should ONLY optimize when:
;   1. Base is a CONSTANT that needs repeated materialization
;   2. Multiple patterns exist to chain together (single pattern in loop is skipped)
;
; The pass should SKIP when:
;   1. Base is a non-constant SSA value (already shared, no materialization savings)
;   2. Single pattern inside a loop (creates counterproductive phi nodes)
;
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=amdgpu-tdm-optimization -S < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

declare void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32>, <8 x i32>, i32 immarg)
declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.amdgcn.workitem.id.x()

;===----------------------------------------------------------------------===;
; CASE 1: Constant Base + Multiple Patterns → SHOULD OPTIMIZE
;         Tests both address descriptors (4xi32) and tensor descriptors (8xi32)
;===----------------------------------------------------------------------===;

; Address descriptors with constant base - SHOULD OPTIMIZE
; CHECK-LABEL: @opt_addr_constant_base_multi
; CHECK: %tdm_desc_storage = alloca <4 x i32>
; CHECK: store <4 x i32> <i32 1,
define amdgpu_kernel void @opt_addr_constant_base_multi(i32 %val1, i32 %val2, i32 %val3, i32 %val4) {
entry:
  ; Two address descriptors from constant base
  %addr1.0 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %val1, i64 1
  %addr1.1 = insertelement <4 x i32> %addr1.0, i32 %val2, i64 2
  %addr1   = insertelement <4 x i32> %addr1.1, i32 0, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %addr1, <8 x i32> zeroinitializer, i32 0)

  %addr2.0 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %val3, i64 1
  %addr2.1 = insertelement <4 x i32> %addr2.0, i32 %val4, i64 2
  %addr2   = insertelement <4 x i32> %addr2.1, i32 0, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %addr2, <8 x i32> zeroinitializer, i32 0)
  ret void
}

; Tensor descriptors with constant base - SHOULD OPTIMIZE
; CHECK-LABEL: @opt_tensor_constant_base_multi
; CHECK: %tdm_desc_storage = alloca <8 x i32>
; CHECK: store <8 x i32> <i32 122683392,
define amdgpu_kernel void @opt_tensor_constant_base_multi(i32 %val1, i32 %val2, i32 %val3, i32 %val4) {
entry:
  ; Two tensor descriptors from constant base
  %tensor1.0 = insertelement <8 x i32> <i32 122683392, i32 poison, i32 poison, i32 8388608, i32 16, i32 128, i32 0, i32 0>, i32 %val1, i64 1
  %tensor1   = insertelement <8 x i32> %tensor1.0, i32 %val2, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> zeroinitializer, <8 x i32> %tensor1, i32 0)

  %tensor2.0 = insertelement <8 x i32> <i32 122683392, i32 poison, i32 poison, i32 8388608, i32 16, i32 128, i32 0, i32 0>, i32 %val3, i64 1
  %tensor2   = insertelement <8 x i32> %tensor2.0, i32 %val4, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> zeroinitializer, <8 x i32> %tensor2, i32 0)
  ret void
}

;===----------------------------------------------------------------------===;
; CASE 2: SSA Base + Multiple Patterns → SHOULD SKIP
;         Tests both address descriptors (4xi32) and tensor descriptors (8xi32)
;===----------------------------------------------------------------------===;

; Address descriptors with SSA base (insertelement result) - SHOULD SKIP
; CHECK-LABEL: @skip_addr_ssa_base_multi
; CHECK-NOT: alloca <4 x i32>
; CHECK: insertelement <4 x i32> %addr_base
; CHECK: insertelement <4 x i32> %addr_base
define amdgpu_kernel void @skip_addr_ssa_base_multi(i32 %flag, i32 %val1, i32 %val2, i32 %val3) {
entry:
  ; Create SSA base via insertelement (not a constant)
  %addr_base = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %flag, i64 0

  ; Two address descriptors from SSA base - should NOT optimize
  %addr1 = insertelement <4 x i32> %addr_base, i32 %val1, i64 1
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %addr1, <8 x i32> zeroinitializer, i32 0)

  %addr2 = insertelement <4 x i32> %addr_base, i32 %val2, i64 1
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %addr2, <8 x i32> zeroinitializer, i32 0)
  ret void
}

; Tensor descriptors with SSA base (shufflevector result) - SHOULD SKIP
; CHECK-LABEL: @skip_tensor_ssa_base_multi
; CHECK-NOT: alloca <8 x i32>
; CHECK: shufflevector
; CHECK: insertelement <8 x i32> %tensor_base
; CHECK: insertelement <8 x i32> %tensor_base
define amdgpu_kernel void @skip_tensor_ssa_base_multi(i32 %n, i32 %val1, i32 %val2) {
entry:
  ; Create SSA base via shufflevector (not a constant)
  %base_dim0 = shl i32 %n, 16
  %tensor_base.0 = insertelement <8 x i32> <i32 122683392, i32 poison, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>, i32 %base_dim0, i64 1
  %tensor_base = shufflevector <8 x i32> %tensor_base.0, <8 x i32> <i32 poison, i32 poison, i32 poison, i32 8388608, i32 16, i32 128, i32 0, i32 0>, <8 x i32> <i32 0, i32 1, i32 poison, i32 11, i32 12, i32 13, i32 14, i32 15>

  ; Two tensor descriptors from SSA base - should NOT optimize
  %tensor1 = insertelement <8 x i32> %tensor_base, i32 %val1, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> zeroinitializer, <8 x i32> %tensor1, i32 0)

  %tensor2 = insertelement <8 x i32> %tensor_base, i32 %val2, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> zeroinitializer, <8 x i32> %tensor2, i32 0)
  ret void
}

;===----------------------------------------------------------------------===;
; CASE 3: Loop + Constant Base + Single Pattern → SHOULD SKIP
;         Single pattern in loop creates counterproductive phi nodes
;===----------------------------------------------------------------------===;

; Single address descriptor in loop - SHOULD SKIP
; CHECK-LABEL: @skip_addr_loop_single
; CHECK-NOT: alloca
; CHECK: insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>
define amdgpu_kernel void @skip_addr_loop_single(i32 %n, i32 %lds_base) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]

  ; Compute loop-varying offset
  %k_offset = mul i32 %i, 128
  %lds_offset = add i32 %lds_base, %k_offset

  ; Single address descriptor from constant base - all fields loop-dependent
  %addr.0 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %lds_offset, i64 1
  %addr.1 = insertelement <4 x i32> %addr.0, i32 %k_offset, i64 2
  %addr   = insertelement <4 x i32> %addr.1, i32 0, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %addr, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.barrier()

  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

; Single tensor descriptor in loop - SHOULD SKIP
; CHECK-LABEL: @skip_tensor_loop_single
; CHECK-NOT: alloca
; CHECK: insertelement <8 x i32> <i32 122683392
define amdgpu_kernel void @skip_tensor_loop_single(i32 %n, i32 %base_dim) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]

  ; Compute loop-varying dimensions
  %k_offset = mul i32 %i, 128
  %k_remaining = sub i32 %n, %k_offset
  %dim1 = shl i32 %k_remaining, 16
  %dim1_or = or i32 %dim1, %base_dim

  ; Single tensor descriptor from constant base - all fields loop-dependent
  %tensor.0 = insertelement <8 x i32> <i32 122683392, i32 poison, i32 poison, i32 8388608, i32 16, i32 128, i32 0, i32 0>, i32 %dim1_or, i64 1
  %tensor   = insertelement <8 x i32> %tensor.0, i32 %k_remaining, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> zeroinitializer, <8 x i32> %tensor, i32 0)
  call void @llvm.amdgcn.s.barrier()

  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

;===----------------------------------------------------------------------===;
; CASE 4: Loop + Constant Base + Multiple Patterns → SHOULD OPTIMIZE
;         Multiple patterns in loop benefit from chaining
;===----------------------------------------------------------------------===;

; Multiple address descriptors in loop with constant base - SHOULD OPTIMIZE
; CHECK-LABEL: @opt_addr_loop_multi
; CHECK: %tdm_desc_storage = alloca <4 x i32>
; CHECK: store <4 x i32> <i32 1,
define amdgpu_kernel void @opt_addr_loop_multi(i32 %n, i32 %lds_base) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]

  ; Compute loop-varying offsets
  %k_offset = mul i32 %i, 128
  %lds_offset1 = add i32 %lds_base, %k_offset
  %lds_offset2 = add i32 %lds_offset1, 8192

  ; Two address descriptors from constant base
  %addr1.0 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %lds_offset1, i64 1
  %addr1.1 = insertelement <4 x i32> %addr1.0, i32 %k_offset, i64 2
  %addr1   = insertelement <4 x i32> %addr1.1, i32 0, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %addr1, <8 x i32> zeroinitializer, i32 0)

  %addr2.0 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %lds_offset2, i64 1
  %addr2.1 = insertelement <4 x i32> %addr2.0, i32 %k_offset, i64 2
  %addr2   = insertelement <4 x i32> %addr2.1, i32 0, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %addr2, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.barrier()

  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

; Multiple tensor descriptors in loop with constant base - SHOULD OPTIMIZE
; CHECK-LABEL: @opt_tensor_loop_multi
; CHECK: %tdm_desc_storage = alloca <8 x i32>
; CHECK: store <8 x i32> <i32 122683392,
define amdgpu_kernel void @opt_tensor_loop_multi(i32 %n, i32 %base_dim) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]

  ; Compute loop-varying dimensions
  %k_offset = mul i32 %i, 128
  %k_remaining1 = sub i32 %n, %k_offset
  %k_remaining2 = sub i32 %k_remaining1, 64
  %dim1 = shl i32 %k_remaining1, 16
  %dim1_or = or i32 %dim1, %base_dim
  %dim2 = shl i32 %k_remaining2, 16
  %dim2_or = or i32 %dim2, %base_dim

  ; Two tensor descriptors from constant base
  %tensor1.0 = insertelement <8 x i32> <i32 122683392, i32 poison, i32 poison, i32 8388608, i32 16, i32 128, i32 0, i32 0>, i32 %dim1_or, i64 1
  %tensor1   = insertelement <8 x i32> %tensor1.0, i32 %k_remaining1, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> zeroinitializer, <8 x i32> %tensor1, i32 0)

  %tensor2.0 = insertelement <8 x i32> <i32 122683392, i32 poison, i32 poison, i32 8388608, i32 16, i32 128, i32 0, i32 0>, i32 %dim2_or, i64 1
  %tensor2   = insertelement <8 x i32> %tensor2.0, i32 %k_remaining2, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> zeroinitializer, <8 x i32> %tensor2, i32 0)
  call void @llvm.amdgcn.s.barrier()

  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

;===----------------------------------------------------------------------===;
; Negative Tests: Patterns that should NOT be optimized
;===----------------------------------------------------------------------===;

; All fields variable, no constants - low benefit, should not optimize
; CHECK-LABEL: @test_no_opt_low_reuse(
; CHECK-NOT: alloca
; CHECK: insertelement <4 x i32> <i32 poison
define amdgpu_kernel void @test_no_opt_low_reuse(i32 %val1, i32 %val2, i32 %val3, i32 %val4) {
entry:
  %insert1 = insertelement <4 x i32> <i32 poison, i32 poison, i32 poison, i32 poison>, i32 %val1, i64 0
  %insert2 = insertelement <4 x i32> %insert1, i32 %val2, i64 1
  %insert3 = insertelement <4 x i32> %insert2, i32 %val3, i64 2
  %insert4 = insertelement <4 x i32> %insert3, i32 %val4, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %insert4, <8 x i32> zeroinitializer, i32 0)
  ret void
}

;===----------------------------------------------------------------------===;
; Threshold Testing: Verify different benefit scores work with threshold
;===----------------------------------------------------------------------===;

; Test: Low benefit (should not optimize with high threshold=50)
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=amdgpu-tdm-optimization -amdgpu-tdm-opt-threshold=50 -S < %s | FileCheck %s --check-prefix=THRESHOLD50
; THRESHOLD50-LABEL: @test_threshold_high_blocks_opt(
; THRESHOLD50-NOT: alloca
; THRESHOLD50: insertelement
define amdgpu_kernel void @test_threshold_high_blocks_opt(i32 %val1, i32 %val2) {
entry:
  ; Two patterns with 3 constant fields each = benefit 12, below threshold=50
  %desc1 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %val1, i64 1
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %desc1, <8 x i32> zeroinitializer, i32 0)

  %desc2 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %val2, i64 1
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %desc2, <8 x i32> zeroinitializer, i32 0)
  ret void
}

; Test: Same pattern should optimize with lower threshold=2
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=amdgpu-tdm-optimization -amdgpu-tdm-opt-threshold=2 -S < %s | FileCheck %s --check-prefix=THRESHOLD2
; THRESHOLD2-LABEL: @test_threshold_low_allows_opt(
; THRESHOLD2: alloca <4 x i32>
; THRESHOLD2: store <4 x i32> <i32 1,
define amdgpu_kernel void @test_threshold_low_allows_opt(i32 %val1, i32 %val2) {
entry:
  ; Two patterns - benefit 12, above threshold=2
  %desc1 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %val1, i64 1
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %desc1, <8 x i32> zeroinitializer, i32 0)

  %desc2 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %val2, i64 1
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %desc2, <8 x i32> zeroinitializer, i32 0)
  ret void
}

; Test: Loop with MULTIPLE patterns gets loop multiplier benefit
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=amdgpu-tdm-optimization -amdgpu-tdm-opt-threshold=20 -S < %s | FileCheck %s --check-prefix=THRESHOLD-LOOP
; THRESHOLD-LOOP-LABEL: @test_threshold_loop_multiplier(
; THRESHOLD-LOOP: alloca <4 x i32>
define amdgpu_kernel void @test_threshold_loop_multiplier(ptr %data, i32 %count) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %ptr = getelementptr i32, ptr %data, i32 %i
  %val = load i32, ptr %ptr

  ; Two patterns in loop: benefit = 2 * 3 * 2 * 5 = 60 (above threshold=20)
  %desc1 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %val, i64 1
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %desc1, <8 x i32> zeroinitializer, i32 0)

  %val2 = add i32 %val, 100
  %desc2 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %val2, i64 1
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %desc2, <8 x i32> zeroinitializer, i32 0)

  %next = add i32 %i, 1
  %cond = icmp ult i32 %next, %count
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

;===----------------------------------------------------------------------===;
; Cleanup Tests: Verify that insertelement chains are removed after transformation
;===----------------------------------------------------------------------===;

; Verify cleanup of address descriptor chains
; CHECK-LABEL: @test_cleanup_address_descriptor(
; CHECK: %tdm_desc_storage = alloca <4 x i32>
; CHECK: store <4 x i32> <i32 1,
; CHECK-NOT: insertelement <4 x i32> <i32 1, i32 poison
define amdgpu_kernel void @test_cleanup_address_descriptor(i32 %val1, i32 %val2, i32 %val3, i32 %val4) {
entry:
  ; First pattern - chain should be removed
  %insert1 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %val1, i64 1
  %insert2 = insertelement <4 x i32> %insert1, i32 %val2, i64 2
  %insert3 = insertelement <4 x i32> %insert2, i32 %val3, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %insert3, <8 x i32> zeroinitializer, i32 0)

  ; Second pattern - also removed
  %insert4 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %val4, i64 1
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %insert4, <8 x i32> zeroinitializer, i32 0)
  ret void
}

; Verify cleanup with multiple patterns sharing storage
; CHECK-LABEL: @test_cleanup_multiple_patterns(
; CHECK: %tdm_desc_storage = alloca <4 x i32>
; CHECK-NOT: alloca <4 x i32>
; CHECK: load <4 x i32>, ptr addrspace(5) %tdm_desc_storage
; CHECK: call void @llvm.amdgcn.tensor.load.to.lds.d2
; CHECK: load <4 x i32>, ptr addrspace(5) %tdm_desc_storage
; CHECK: call void @llvm.amdgcn.tensor.load.to.lds.d2
define amdgpu_kernel void @test_cleanup_multiple_patterns(i32 %a1, i32 %a2, i32 %b1, i32 %b2) {
entry:
  ; First pattern
  %insert1a = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %a1, i64 1
  %insert2a = insertelement <4 x i32> %insert1a, i32 %a2, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %insert2a, <8 x i32> zeroinitializer, i32 0)

  ; Second pattern (shares base template with first)
  %insert1b = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %b1, i64 1
  %insert2b = insertelement <4 x i32> %insert1b, i32 %b2, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %insert2b, <8 x i32> zeroinitializer, i32 0)
  ret void
}
