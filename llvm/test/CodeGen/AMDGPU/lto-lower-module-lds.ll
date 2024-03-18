
; Default O0
; RUN: opt -mtriple=amdgcn-- -mcpu=gfx1030 %s -o %t.bc
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O0
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgcn-- -mcpu=gfx1030 %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O1
; RUN: opt -mtriple=amdgcn-- -mcpu=gfx1030 %s -o %t.bc
; RUN: llvm-lto2 run -O1 -cg-opt-level 1 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O1
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgcn-- -mcpu=gfx1030 %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O1 -cg-opt-level 1 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O2
; RUN: opt -mtriple=amdgcn-- -mcpu=gfx1030 %s -o %t.bc
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O2
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgcn-- -mcpu=gfx1030 %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O3
; RUN: opt -mtriple=amdgcn-- -mcpu=gfx1030 %s -o %t.bc
; RUN: llvm-lto2 run -O3 -cg-opt-level 3 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O3
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgcn-- -mcpu=gfx1030 %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O3 -cg-opt-level 3 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; First print will be from the New PM during the full LTO pipeline.
; Second print will be from the legacy PM during the CG pipeline.

; CHECK: Running pass: AMDGPULowerModuleLDSPass on [module]
; CHECK: ModulePass Manager
; CHECK:   Lower uses of LDS variables from non-kernel functions

@lds = internal unnamed_addr addrspace(3) global i32 undef, align 4

define amdgpu_kernel void @test() {
entry:
  store i32 1, ptr addrspace(3) @lds
  ret void
}
