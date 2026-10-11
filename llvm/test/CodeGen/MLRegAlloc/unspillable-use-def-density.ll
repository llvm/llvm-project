; REQUIRES: x86_64-linux, amdgpu-registered-target
; RUN: rm -rf %t.rundir
; RUN: rm -rf %t.channel-basename.*
; RUN: mkdir %t.rundir
; RUN: cp %S/../../../lib/Analysis/models/log_reader.py %t.rundir
; RUN: cp %S/../../../lib/Analysis/models/interactive_host.py %t.rundir
; RUN: cp %S/Inputs/interactive_main.py %t.rundir
; RUN: %python %t.rundir/interactive_main.py %t.channel-basename \
; RUN:    llc -mtriple=amdgpu9.08-amd-amdhsa -sgpr-regalloc=fast -wwm-regalloc=fast \
; RUN:    -regalloc-enable-advisor=release -interactive-model-runner-echo-reply \
; RUN:    -regalloc-evict-interactive-channel-base=%t.channel-basename %s -o /dev/null | \
; RUN:    FileCheck %s --implicit-check-not=nan

;; Unspillable live ranges have infinite weight. Their use_def_density is
;; normalized to 1.0 and does not affect the normalization of the others.
;; The fast SGPR and WWM allocators keep greedy to a single context.

; CHECK:      observation: 0
; CHECK-NEXT: mask: 1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
; CHECK:      use_def_density: 1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0

;; Evicting an equal cascade asserts in greedy.
; RUN: %python %t.rundir/interactive_main.py %t.channel-basename.cascade \
; RUN:    llc -mtriple=amdgpu9.00 -start-before=greedy,0 -stop-after=virt-reg-rewriter,2 \
; RUN:    -regalloc-enable-advisor=release -mlregalloc-num-allocatable-regs=256 \
; RUN:    -regalloc-evict-interactive-channel-base=%t.channel-basename.cascade \
; RUN:    %S/../AMDGPU/illegal-eviction-assert.mir -o /dev/null 2>&1 | \
; RUN:    FileCheck %s --check-prefix=CASCADE
; CASCADE: error: <unknown>:0:0: ran out of registers during register allocation

define amdgpu_kernel void @test_spill_av_class(<4 x i32> %arg) #0 {
  %v0 = call i32 asm sideeffect "; def $0", "=v"()
  %tmp = insertelement <2 x i32> poison, i32 %v0, i32 0
  %mai = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %arg, i32 0, i32 0, i32 0)
  store volatile <4 x i32> %mai, ptr addrspace(1) poison
  call void asm sideeffect "; use $0", "v"(<2 x i32> %tmp)
  ret void
}

declare <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32, i32, <4 x i32>, i32, i32, i32)

attributes #0 = { nounwind "amdgpu-num-vgpr"="5" }
