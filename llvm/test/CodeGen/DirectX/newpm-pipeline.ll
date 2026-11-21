; RUN: llc -enable-new-pm -mtriple=dxil-pc-shadermodel6.3-library -O0 -filetype=asm -print-pipeline-passes=tree < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SCALARIZE,ASM
; RUN: llc -enable-new-pm -mtriple=dxil-pc-shadermodel6.3-library -O3 -filetype=asm -print-pipeline-passes=tree < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,OPT,SCALARIZE,ASM
; RUN: llc -enable-new-pm -mtriple=dxil-pc-shadermodel6.10-library -O3 -filetype=obj -print-pipeline-passes=tree < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,OPT,OBJ

; CHECK: require<MachineModuleAnalysis>
; CHECK-NEXT: function
; CHECK-NEXT:   strip-convergence-intrinsics
; CHECK-NEXT: dxil-finalize-linkage
; OPT-NEXT: globaldce
; CHECK-NEXT: dxil-mem-intrinsics
; CHECK-NEXT: dxil-cbuffer-access
; CHECK-NEXT: function
; CHECK-NEXT:   dxil-remove-unused-resources
; CHECK-NEXT:   dxil-resource-access
; CHECK-NEXT: dxil-intrinsic-expansion
; CHECK-NEXT: dxil-data-scalarization
; SCALARIZE-NEXT: function
; SCALARIZE-NEXT:   scalarizer
; CHECK-NEXT: dxil-flatten-arrays
; CHECK-NEXT: function
; CHECK-NEXT:   dxil-forward-handle-accesses
; OPT-NEXT:   dse
; CHECK-NEXT:   dxil-legalize
; CHECK-NEXT: dxil-resource-implicit-binding
; CHECK-NEXT: dxil-translate-metadata
; CHECK-NEXT: dxil-post-optimization-validation
; CHECK-NEXT: dxil-op-lower
; CHECK-NEXT: dxil-prepare-module
; CHECK-NEXT: dxil-debug-info
; ASM-NEXT: dxil-pretty-printer
; OBJ-NEXT: dxil-embedder
; OBJ-NEXT: dxil-container-globals
; OBJ-NEXT: dxil-container-pdb
; OBJ-NEXT: dxil-asm-printer-begin
; OBJ-NEXT: function
; OBJ-NEXT:   machine-function
; OBJ-NEXT:     dxil-asm-printer
; OBJ-NEXT:   free-machine-function
; OBJ-NEXT: dxil-asm-printer-end
