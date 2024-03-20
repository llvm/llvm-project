; REQUIRES: asserts

; RUN: llc -verify-machineinstrs=0 -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -verify-machineinstrs=0 -sgpr-regalloc=greedy -vgpr-regalloc=greedy -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck -check-prefix=DEFAULT %s

; RUN: llc -verify-machineinstrs=0 -O0 -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck -check-prefix=O0 %s

; RUN: llc -verify-machineinstrs=0 -vgpr-regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck -check-prefix=DEFAULT-BASIC %s
; RUN: llc -verify-machineinstrs=0 -sgpr-regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck -check-prefix=BASIC-DEFAULT %s
; RUN: llc -verify-machineinstrs=0 -sgpr-regalloc=basic -vgpr-regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck -check-prefix=BASIC-BASIC %s

; RUN: not --crash llc -verify-machineinstrs=0 -regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck -check-prefix=REGALLOC %s
; RUN: not --crash llc -verify-machineinstrs=0 -regalloc=fast -O0 -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck -check-prefix=REGALLOC %s


; REGALLOC: -regalloc not supported with amdgcn. Use -sgpr-regalloc and -vgpr-regalloc

; DEFAULT: Greedy Register Allocator
; DEFAULT-NEXT: Virtual Register Rewriter
; DEFAULT-NEXT: SI lower SGPR spill instructions
; DEFAULT-NEXT: Virtual Register Map
; DEFAULT-NEXT: Live Register Matrix
; DEFAULT-NEXT: SI Pre-allocate WWM Registers
; DEFAULT-NEXT: Greedy Register Allocator
; DEFAULT-NEXT: SI Lower WWM Copies
; DEFAULT-NEXT: GCN NSA Reassign
; DEFAULT-NEXT: Virtual Register Rewriter
; DEFAULT-NEXT: AMDGPU Mark Last Scratch Load
; DEFAULT-NEXT: Stack Slot Coloring

; O0: Fast Register Allocator
; O0-NEXT: SI lower SGPR spill instructions
; O0-NEXT: Slot index numbering
; O0-NEXT: Live Interval Analysis
; O0-NEXT: Virtual Register Map
; O0-NEXT: Live Register Matrix
; O0-NEXT: SI Pre-allocate WWM Registers
; O0-NEXT: Fast Register Allocator
; O0-NEXT: SI Lower WWM Copies
; O0-NEXT: SI Fix VGPR copies




; BASIC-DEFAULT: Debug Variable Analysis
; BASIC-DEFAULT-NEXT: Live Stack Slot Analysis
; BASIC-DEFAULT-NEXT: Machine Natural Loop Construction
; BASIC-DEFAULT-NEXT: Machine Block Frequency Analysis
; BASIC-DEFAULT-NEXT: Virtual Register Map
; BASIC-DEFAULT-NEXT: Live Register Matrix
; BASIC-DEFAULT-NEXT: Basic Register Allocator
; BASIC-DEFAULT-NEXT: Virtual Register Rewriter
; BASIC-DEFAULT-NEXT: SI lower SGPR spill instructions
; BASIC-DEFAULT-NEXT: Virtual Register Map
; BASIC-DEFAULT-NEXT: Live Register Matrix
; BASIC-DEFAULT-NEXT: SI Pre-allocate WWM Registers
; BASIC-DEFAULT-NEXT: Bundle Machine CFG Edges
; BASIC-DEFAULT-NEXT: Spill Code Placement Analysis
; BASIC-DEFAULT-NEXT: Lazy Machine Block Frequency Analysis
; BASIC-DEFAULT-NEXT: Machine Optimization Remark Emitter
; BASIC-DEFAULT-NEXT: Greedy Register Allocator
; BASIC-DEFAULT-NEXT: SI Lower WWM Copies
; BASIC-DEFAULT-NEXT: GCN NSA Reassign
; BASIC-DEFAULT-NEXT: Virtual Register Rewriter
; BASIC-DEFAULT-NEXT: AMDGPU Mark Last Scratch Load
; BASIC-DEFAULT-NEXT: Stack Slot Coloring



; DEFAULT-BASIC: Greedy Register Allocator
; DEFAULT-BASIC-NEXT: Virtual Register Rewriter
; DEFAULT-BASIC-NEXT: SI lower SGPR spill instructions
; DEFAULT-BASIC-NEXT: Virtual Register Map
; DEFAULT-BASIC-NEXT: Live Register Matrix
; DEFAULT-BASIC-NEXT: SI Pre-allocate WWM Registers
; DEFAULT-BASIC-NEXT: Basic Register Allocator
; DEFAULT-BASIC-NEXT: SI Lower WWM Copies
; DEFAULT-BASIC-NEXT: GCN NSA Reassign
; DEFAULT-BASIC-NEXT: Virtual Register Rewriter
; DEFAULT-BASIC-NEXT: AMDGPU Mark Last Scratch Load
; DEFAULT-BASIC-NEXT: Stack Slot Coloring



; BASIC-BASIC: Debug Variable Analysis
; BASIC-BASIC-NEXT: Live Stack Slot Analysis
; BASIC-BASIC-NEXT: Machine Natural Loop Construction
; BASIC-BASIC-NEXT: Machine Block Frequency Analysis
; BASIC-BASIC-NEXT: Virtual Register Map
; BASIC-BASIC-NEXT: Live Register Matrix
; BASIC-BASIC-NEXT: Basic Register Allocator
; BASIC-BASIC-NEXT: Virtual Register Rewriter
; BASIC-BASIC-NEXT: SI lower SGPR spill instructions
; BASIC-BASIC-NEXT: Virtual Register Map
; BASIC-BASIC-NEXT: Live Register Matrix
; BASIC-BASIC-NEXT: SI Pre-allocate WWM Registers
; BASIC-BASIC-NEXT: Basic Register Allocator
; BASIC-BASIC-NEXT: SI Lower WWM Copies
; BASIC-BASIC-NEXT: GCN NSA Reassign
; BASIC-BASIC-NEXT: Virtual Register Rewriter
; BASIC-BASIC-NEXT: AMDGPU Mark Last Scratch Load
; BASIC-BASIC-NEXT: Stack Slot Coloring


declare void @bar()

; Something with some CSR SGPR spills
define void @foo() {
  call void asm sideeffect "; clobber", "~{s33}"()
  call void @bar()
  ret void
}

; Block live out spills with fast regalloc
define amdgpu_kernel void @control_flow(i1 %cond) {
  %s33 = call i32 asm sideeffect "; clobber", "={s33}"()
  br i1 %cond, label %bb0, label %bb1

bb0:
   call void asm sideeffect "; use %0", "s"(i32 %s33)
   br label %bb1

bb1:
  ret void
}
