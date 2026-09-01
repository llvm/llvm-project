; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 \
; RUN:   -stop-after=machine-cse -o - %s | FileCheck %s --check-prefix=CSE
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 \
; RUN:   -stop-after=early-machinelicm -o - %s | FileCheck %s --check-prefix=LICM
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 \
; RUN:   -stop-after=register-coalescer -o - %s | FileCheck %s --check-prefix=COALESCE
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 \
; RUN:   -stop-after=dead-mi-elimination -o - %s | FileCheck %s --check-prefix=DCE
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 -global-isel=1 \
; RUN:   -global-isel-abort=1 -stop-after=dead-mi-elimination -o - %s \
; RUN:   | FileCheck %s --check-prefix=DCE
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 \
; RUN:   -stop-after=si-optimize-exec-masking-pre-ra -o - %s \
; RUN:   | FileCheck %s --check-prefix=PRERA
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 -global-isel=1 \
; RUN:   -global-isel-abort=1 -stop-after=si-optimize-exec-masking-pre-ra \
; RUN:   -o - %s | FileCheck %s --check-prefix=PRERA
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O0 \
; RUN:   -stop-after=finalize-isel -o - %s \
; RUN:   | llc -x=mir -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
; RUN:       -run-pass=machine-cse -verify-machineinstrs -o /dev/null -

define amdgpu_kernel void @preserve_two_handoffs(ptr addrspace(1) %out,
                                                  i32 %value) {
; CSE-LABEL: name: preserve_two_handoffs
; CSE: nomerge REGALLOC_HANDOFF_VGPR
; CSE: nomerge REGALLOC_HANDOFF_VGPR
; COALESCE-LABEL: name: preserve_two_handoffs
; COALESCE: nomerge REGALLOC_HANDOFF_VGPR
; COALESCE: nomerge REGALLOC_HANDOFF_VGPR
  %first = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  %second = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  store volatile i32 %first, ptr addrspace(1) %out
  %next = getelementptr i32, ptr addrspace(1) %out, i32 1
  store volatile i32 %second, ptr addrspace(1) %next
  ret void
}

define amdgpu_kernel void @preserve_loop_handoffs(ptr addrspace(1) %out,
                                                   i32 %value, i32 %count) {
; CSE-LABEL: name: preserve_loop_handoffs
; COALESCE-LABEL: name: preserve_loop_handoffs
; LICM-LABEL: name: preserve_loop_handoffs
; LICM-NOT: REGALLOC_HANDOFF_
; LICM: bb.3.loop:
; LICM: nomerge REGALLOC_HANDOFF_VGPR
; LICM: nomerge REGALLOC_HANDOFF_VGPR
entry:
  %positive = icmp sgt i32 %count, 0
  br i1 %positive, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %first = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  %second = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  %sum = add i32 %first, %second
  %ptr = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store volatile i32 %sum, ptr addrspace(1) %ptr
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define amdgpu_kernel void @preserve_unused_handoff(i32 %value) {
; DCE-LABEL: name: preserve_unused_handoff
; DCE: nomerge REGALLOC_HANDOFF_VGPR
; PRERA-LABEL: name: preserve_unused_handoff
; PRERA: nomerge REGALLOC_HANDOFF_VGPR
  %unused = call i32 @llvm.experimental.regalloc.handoff(
      i32 %value, metadata !0)
  ret void
}

declare i32 @llvm.experimental.regalloc.handoff(i32, metadata)

!0 = !{!"amdgpu.vgpr"}
