; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1151 -mattr=-real-true16 -O3 -amdgpu-stress-sgpr=16 -verify-machineinstrs -stop-after=greedy %s -o - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1151 -mattr=-real-true16 -O3 -amdgpu-stress-sgpr=16 -verify-machineinstrs %s -o /dev/null
;
; Reduced from reduced_triton380_gfx1151.ll, a Triton 3.8.0 attention kernel
; (issue #225054). Limit SGPRs to preserve the original register pressure with
; fewer instructions. Greedy splits the buffer descriptor: a spilled sibling
; only holds sub2/sub3, but shares a stack slot with the complete descriptor.
; Storing that sibling full-width would overwrite the live sub0/sub1 (the
; buffer address) with undef. The loop reloads and uses all four words on the
; next iteration, so those low words must survive the partial store.
;
; Check the shared slot, the high-only split def, its exact byte offsets, and
; the backedge to the full descriptor load. No other store may repair the slot
; between the partial store and the backedge.
;
; CHECK-LABEL: name: partial_spill_preserves_descriptor
; CHECK: SI_SPILL_S128_SAVE {{%[0-9]+}}, [[SLOT:%stack\.[0-9]+]],
; CHECK: bb.[[LOOP:[0-9]+]].loop:
; CHECK-NEXT: successors: %bb.[[LOOP]]
; CHECK-NOT: SI_SPILL_S128_SAVE {{.*}}[[SLOT]],
; CHECK: [[DESC:%[0-9]+]]:sgpr_128 = SI_SPILL_S128_RESTORE [[SLOT]],
; CHECK-NEXT: {{%[0-9]+}}:vreg_128 = BUFFER_LOAD_DWORDX4_OFFEN {{%[0-9]+}}, [[DESC]],
; CHECK-NEXT: undef [[HIGH:%[0-9]+]].sub2_sub3:sgpr_128 = lr-split COPY [[DESC]].sub2_sub3
; CHECK-NOT: SI_SPILL_S128_SAVE {{.*}}[[SLOT]],
; CHECK: undef [[SPILL:%[0-9]+]].sub2_sub3:sgpr_128 = lr-split COPY [[HIGH]].sub2_sub3
; CHECK-NEXT: SI_SPILL_S32_SAVE_PARTIAL [[SPILL]].sub2, [[SLOT]], 2, {{.*}}(store (s32) into [[SLOT]] + 8, addrspace 5)
; CHECK-NEXT: SI_SPILL_S32_SAVE_PARTIAL [[SPILL]].sub3, [[SLOT]], 3, {{.*}}(store (s32) into [[SLOT]] + 12, addrspace 5)
; CHECK-NOT: {{SI_SPILL_.*SAVE.*}} [[SLOT]],
; CHECK: S_CBRANCH_VCCNZ %bb.[[LOOP]],

target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @partial_spill_preserves_descriptor(ptr addrspace(1) %other, ptr addrspace(1) %base, ptr addrspace(1) %tail) {
entry:
  %step = load i32, ptr addrspace(1) %other, align 4
  br label %loop

loop:
  %offset0 = phi i32 [ 1, %entry ], [ 0, %loop ]
  %offset1 = phi i32 [ 52, %entry ], [ 0, %loop ]
  %offset2 = phi i32 [ 53, %entry ], [ 0, %loop ]
  %offset3 = phi i32 [ 54, %entry ], [ 0, %loop ]
  %offset4 = phi i32 [ 56, %entry ], [ 0, %loop ]
  %offset5 = phi i32 [ 57, %entry ], [ 0, %loop ]
  %offset6 = phi i32 [ 0, %entry ], [ 1, %loop ]
  %offset = phi i32 [ 0, %entry ], [ %step, %loop ]
  %descriptor = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %base, i16 0, i64 2147483646, i32 822243328)
  %negative = icmp slt i32 %offset, 0
  %load.offset = select i1 %negative, i32 0, i32 1
  %data = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %descriptor, i32 %load.offset, i32 0, i32 0)
  store <4 x i32> %data, ptr addrspace(3) null, align 16
  %v18 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> zeroinitializer, <16 x i16> splat (i16 1), <8 x float> zeroinitializer)
  %v19 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> zeroinitializer, <16 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 16256, i16 16256, i16 16256, i16 16256, i16 16256, i16 16256, i16 16256, i16 16256>, <8 x float> %v18)
  %v20 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> zeroinitializer, <16 x i16> <i16 1, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, <8 x float> %v19)
  %v21 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> zeroinitializer, <16 x i16> splat (i16 32704), <8 x float> %v20)
  %v22 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> zeroinitializer, <16 x i16> splat (i16 16256), <8 x float> %v21)
  %v23 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> <i16 16256, i16 16256, i16 16256, i16 16256, i16 16256, i16 16256, i16 16256, i16 16256, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, <16 x i16> zeroinitializer, <8 x float> %v22)
  %v24 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> zeroinitializer, <16 x i16> zeroinitializer, <8 x float> %v23)
  %v25 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> zeroinitializer, <16 x i16> zeroinitializer, <8 x float> %v24)
  %other.descriptor = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %other, i16 0, i64 2147483646, i32 0)
  %v27 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %other.descriptor, i32 %offset0, i32 0, i32 0)
  %v28 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %other.descriptor, i32 %offset1, i32 0, i32 0)
  %v29 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %other.descriptor, i32 %offset2, i32 0, i32 0)
  %v30 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %other.descriptor, i32 %offset3, i32 0, i32 0)
  %v31 = shl i32 %offset4, 1
  %v32 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %other.descriptor, i32 %v31, i32 0, i32 0)
  %v33 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %other.descriptor, i32 %offset6, i32 0, i32 0)
  %v34 = insertelement <8 x i16> zeroinitializer, i16 %v27, i64 0
  store <8 x i16> %v34, ptr addrspace(3) null, align 16
  store <8 x i16> zeroinitializer, ptr addrspace(3) inttoptr (i32 16 to ptr addrspace(3)), align 16
  store <8 x i16> zeroinitializer, ptr addrspace(3) null, align 16
  %v35 = insertelement <8 x i16> zeroinitializer, i16 %v28, i64 2
  %v36 = insertelement <8 x i16> %v35, i16 %v30, i64 3
  %v37 = insertelement <8 x i16> %v36, i16 %v32, i64 1
  %v38 = insertelement <8 x i16> %v37, i16 %v33, i64 0
  store <8 x i16> %v38, ptr addrspace(3) inttoptr (i32 16 to ptr addrspace(3)), align 16
  %v39 = insertelement <8 x i16> splat (i16 1), i16 %v29, i64 0
  %v40 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %other.descriptor, i32 %offset5, i32 0, i32 0)
  %v41 = insertelement <8 x i16> %v39, i16 %v40, i64 1
  store <8 x i16> %v41, ptr addrspace(3) null, align 16
  fence release
  %v42 = shufflevector <8 x float> %v25, <8 x float> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x float> %v42, ptr addrspace(3) null, align 16
  store <4 x float> zeroinitializer, ptr addrspace(3) inttoptr (i32 512 to ptr addrspace(3)), align 16
  %tail.descriptor = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %tail, i16 0, i64 2147483646, i32 822243328)
  %v44 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %tail.descriptor, i32 %offset, i32 0, i32 0)
  %v45 = bitcast <4 x i32> %v44 to <8 x bfloat>
  %v46 = extractelement <8 x bfloat> %v45, i64 0
  %v47 = insertelement <4 x bfloat> zeroinitializer, bfloat %v46, i64 0
  store <4 x bfloat> %v47, ptr addrspace(3) null, align 8
  br label %loop
}

declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8), i32, i32, i32 immarg)

declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16>, <16 x i16>, <8 x float>)

declare i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8), i32, i32, i32 immarg)

declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1), i16, i64, i32)
