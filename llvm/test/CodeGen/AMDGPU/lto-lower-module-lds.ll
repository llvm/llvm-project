
; Default O0
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O0
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O1
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O1 -cg-opt-level 1 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O1
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O1 -cg-opt-level 1 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O2
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O2
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O3
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O3 -cg-opt-level 3 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O3
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O3 -cg-opt-level 3 %t.bc -o %t.s -r %t.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; First print will be from the New PM during the full LTO pipeline.
; Second print will be from the legacy PM during the CG pipeline.

; CHECK: Running pass: AMDGPULowerModuleLDSPass on [module]
; CHECK: ModulePass Manager
; CHECK:   Lower uses of LDS variables from non-kernel functions

; Test -enable-npm-for-backend.

; NPM Default O0
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=force-on -debug-pass-manager 2>&1 | FileCheck --check-prefix=NPM %s

; NPM Unified O0
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=force-on -debug-pass-manager 2>&1 | FileCheck --check-prefix=NPM %s

; NPM Default O2
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=force-on -debug-pass-manager 2>&1 | FileCheck --check-prefix=NPM %s

; NPM Unified O2
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=force-on -debug-pass-manager 2>&1 | FileCheck --check-prefix=NPM %s

; The New PM full-LTO pipeline still runs the module-LDS lowering, and the CG
; pipeline is now driven by the New PM (no legacy "ModulePass Manager" structure).
; NPM-NOT: ModulePass Manager
; NPM: Running pass: AMDGPULowerModuleLDSPass on [module]
; NPM: Running pass: SelectionDAGISelPass on test
; NPM: Running pass: PrologEpilogInserterPass on test
; NPM: Running pass: AMDGPUAsmPrinterPass on test

; Test -print-pipeline-passes prints the New PM codegen pipeline.

; PP Default O2
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=force-on -print-pipeline-passes 2>&1 | FileCheck --check-prefix=PP %s

; PP Unified O2
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=force-on -print-pipeline-passes 2>&1 | FileCheck --check-prefix=PP %s

; First line is the full-LTO (IR) pipeline, second line is the codegen (machine)
; pipeline. Check a few codegen-specific passes (in order) on the second line.
; PP: amdgpu-lower-module-lds
; PP: require<MachineModuleAnalysis>
; PP-SAME: amdgpu-isel
; PP-SAME: prolog-epilog
; PP-SAME: amdgpu-asm-printer

; Test -enable-npm-for-backend=force-disable always drives the CodeGen pipeline
; with the legacy PM, regardless of the target default.

; DISABLE Default O0
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=force-disable -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=DISABLE %s

; DISABLE Unified O0
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=force-disable -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=DISABLE %s

; DISABLE Default O2
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=force-disable -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=DISABLE %s

; DISABLE Unified O2
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=force-disable -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=DISABLE %s

; force-disable keeps the legacy CodeGen PM even after the New PM CodeGen
; pipeline becomes the AMDGPU default, so no New PM CodeGen passes should run.
; DISABLE: Running pass: AMDGPULowerModuleLDSPass on [module]
; DISABLE: ModulePass Manager
; DISABLE:   Lower uses of LDS variables from non-kernel functions
; DISABLE-NOT: Running pass: SelectionDAGISelPass

; Test -enable-npm-for-backend=auto follows the target default, which is
; currently the legacy CodeGen PM for AMDGPU. Update the checks below to the New
; PM CodeGen pipeline when it becomes the default for AMDGPU.

; AUTO Default O0
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=auto -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=AUTO %s

; AUTO Unified O0
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=auto -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=AUTO %s

; AUTO Default O2
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=auto -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=AUTO %s

; AUTO Unified O2
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.bc
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.bc -o %t.s -r %t.bc,test,px -enable-npm-for-backend=auto -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck --check-prefix=AUTO %s

; auto currently maps to the legacy CodeGen PM for AMDGPU (target default false).
; AUTO: Running pass: AMDGPULowerModuleLDSPass on [module]
; AUTO: ModulePass Manager
; AUTO:   Lower uses of LDS variables from non-kernel functions
; AUTO-NOT: Running pass: SelectionDAGISelPass

@lds = internal unnamed_addr addrspace(3) global i32 poison, align 4

define amdgpu_kernel void @test() {
entry:
  store i32 1, ptr addrspace(3) @lds
  ret void
}
