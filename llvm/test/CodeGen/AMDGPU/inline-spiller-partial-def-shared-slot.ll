; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-use-amdgpu-trackers -amdgpu-stress-sgpr=20 -verify-machineinstrs -verify-regalloc -stop-after=greedy -o - %s | FileCheck %s
; A partial sibling must preserve live lanes of the shared spill slot.
define amdgpu_kernel void @partial_def_shared_spill_slot(ptr addrspace(1) inreg %p, ptr addrspace(3) %out1, ptr addrspace(3) %out2, i1 %done, <16 x float> %acc, <8 x i1> %mask, <8 x half> %data) {
; CHECK-LABEL: name: partial_def_shared_spill_slot
; CHECK: undef [[PART:%[0-9]+]].sub2:sgpr_128 = lr-split COPY
; CHECK-NOT: SI_SPILL_S128_SAVE [[PART]],
; CHECK: [[MERGE:%[0-9]+]]:sgpr_128 = SI_SPILL_S128_RESTORE %stack.[[SLOT:[0-9]+]],
; CHECK-NEXT: [[MERGE]].sub2:sgpr_128 = lr-split COPY [[PART]].sub2
; CHECK-NEXT: SI_SPILL_S128_SAVE [[MERGE]], %stack.[[SLOT]],
; CHECK: bb.{{[0-9]+}}.exit:
; CHECK: [[FULL:%[0-9]+]]:sgpr_128 = SI_SPILL_S128_RESTORE %stack.[[SLOT]],
; CHECK-NEXT: BUFFER_LOAD_UBYTE_LDS_OFFSET [[FULL]],
entry:
  %desc = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %p, i16 0, i64 0, i32 1)
  br label %loop
loop:
  %pressure = phi <16 x i32> [ %next, %body ], [ zeroinitializer, %entry ]
  br i1 %done, label %exit, label %body
body:
  %next = or <16 x i32> splat (i32 1), %pressure
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %desc, ptr addrspace(3) null, i32 1, i32 0, i32 0, i32 0, i32 0)
  %mfma = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %data, <8 x half> zeroinitializer, <16 x float> %acc, i32 0, i32 0, i32 0)
  store <4 x float> zeroinitializer, ptr addrspace(3) null, align 16
  store <4 x float> zeroinitializer, ptr addrspace(3) %out2, align 16
  %chosen = select <8 x i1> %mask, <8 x float> splat (float +qnan), <8 x float> zeroinitializer
  %slice = shufflevector <8 x float> %chosen, <8 x float> zeroinitializer, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  store <4 x float> %slice, ptr addrspace(3) %out1, align 16
  %scalar = extractelement <16 x float> %mfma, i64 0
  %wide = insertelement <4 x float> zeroinitializer, float %scalar, i64 0
  store <4 x float> %wide, ptr addrspace(3) null, align 16
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) null, ptr addrspace(3) null, i32 1, i32 0, i32 0, i32 0, i32 0)
  br label %loop
exit:
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %desc, ptr addrspace(3) null, i32 1, i32 0, i32 0, i32 0, i32 0)
  ret void
}

declare void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half>, <8 x half>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) readnone, i16, i64, i32)
