
; Default O0
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.default.bc
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O0
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.unified.bc
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O1
; RUN: llvm-lto2 run -O1 -cg-opt-level 1 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O1
; RUN: llvm-lto2 run -unified-lto=full -O1 -cg-opt-level 1 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O2
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O2
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O3
; RUN: llvm-lto2 run -O3 -cg-opt-level 3 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O3
; RUN: llvm-lto2 run -unified-lto=full -O3 -cg-opt-level 3 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; NPM is default for both full LTO and CG pipelines.

; CHECK-NOT: ModulePass Manager
; CHECK: Running pass: AMDGPULowerModuleLDSPass on [module]
; CHECK: Running pass: SelectionDAGISelPass on test
; CHECK: Running pass: PrologEpilogInserterPass on test
; CHECK: Running pass: AMDGPUAsmPrinterPass on test

; Test -force-new-pm-codegen=true.

; ENABLE-NPM Default O0
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -force-new-pm-codegen=true -debug-pass-manager 2>&1 | FileCheck --check-prefix=ENABLE-NPM %s

; ENABLE-NPM Unified O0
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -force-new-pm-codegen=true -debug-pass-manager 2>&1 | FileCheck --check-prefix=ENABLE-NPM %s

; ENABLE-NPM Default O2
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -force-new-pm-codegen=true -debug-pass-manager 2>&1 | FileCheck --check-prefix=ENABLE-NPM %s

; ENABLE-NPM Unified O2
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -force-new-pm-codegen=true -debug-pass-manager 2>&1 | FileCheck --check-prefix=ENABLE-NPM %s

; ENABLE-NPM-NOT: ModulePass Manager
; ENABLE-NPM: Running pass: AMDGPULowerModuleLDSPass on [module]
; ENABLE-NPM: Running pass: SelectionDAGISelPass on test
; ENABLE-NPM: Running pass: PrologEpilogInserterPass on test
; ENABLE-NPM: Running pass: AMDGPUAsmPrinterPass on test

; Test -force-new-pm-codegen=false drives the CodeGen pipeline
; with the legacy PM, regardless of the target default.

; DISABLE-NPM Default O0
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -force-new-pm-codegen=false -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=DISABLE-NPM %s

; DISABLE-NPM Unified O0
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -force-new-pm-codegen=false -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=DISABLE-NPM %s

; DISABLE-NPM Default O2
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -force-new-pm-codegen=false -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=DISABLE-NPM %s

; DISABLE-NPM Unified O2
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -force-new-pm-codegen=false -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=DISABLE-NPM %s

; DISABLE-NPM: Running pass: AMDGPULowerModuleLDSPass on [module]
; DISABLE-NPM: ModulePass Manager
; DISABLE-NPM:   Lower uses of LDS variables from non-kernel functions
; DISABLE-NPM-NOT: Running pass: SelectionDAGISelPass

@lds = internal unnamed_addr addrspace(3) global i32 poison, align 4

define amdgpu_kernel void @test() {
entry:
  store i32 1, ptr addrspace(3) @lds
  ret void
}
