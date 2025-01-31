; RUN: llc -filetype=asm -mtriple=dxil-pc-shadermodel6.3-library -debug-pass=Structure < %s -o /dev/null 2>&1 | grep -v "Verify generated machine code" | FileCheck %s --check-prefixes=CHECK,CHECK-ASM
; RUN: llc -filetype=obj -mtriple=dxil-pc-shadermodel6.3-library -debug-pass=Structure < %s -o /dev/null 2>&1 | grep -v "Verify generated machine code" | FileCheck %s --check-prefixes=CHECK,CHECK-OBJ

; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: DXIL Resource Type Analysis
; CHECK-NEXT: Target Transform Information

; CHECK-OBJ-NEXT: Machine Module Information
; CHECK-OBJ-NEXT: Machine Branch Probability Analysis
; CHECK-OBJ-NEXT: Create Garbage Collector Module Metadata

; CHECK-NEXT: ModulePass Manager
; CHECK-NEXT:   DXIL Finalize Linkage
; CHECK-NEXT:   DXIL Intrinsic Expansion
; CHECK-NEXT:   DXIL Data Scalarization
; CHECK-NEXT:   DXIL Array Flattener
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     DXIL Resource Access
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     Scalarize vector operations
; CHECK-NEXT:   DXIL Resource Binding Analysis
; CHECK-NEXT:   DXIL resource Information
; CHECK-NEXT:   DXIL Shader Flag Analysis
; CHECK-NEXT:   DXIL Module Metadata analysis
; CHECK-NEXT:   DXIL Translate Metadata
; CHECK-NEXT:   DXIL Op Lowering
; CHECK-NEXT:   DXIL Prepare Module

; CHECK-ASM-NEXT: DXIL Metadata Pretty Printer
; CHECK-ASM-NEXT: Print Module IR

; CHECK-OBJ-NEXT: DXIL Embedder
; CHECK-OBJ-NEXT: DXIL Root Signature Analysis
; CHECK-OBJ-NEXT: DXContainer Global Emitter
; CHECK-OBJ-NEXT: FunctionPass Manager
; CHECK-OBJ-NEXT:   Lazy Machine Block Frequency Analysis
; CHECK-OBJ-NEXT:   Machine Optimization Remark Emitter
; CHECK-OBJ-NEXT:   DXIL Assembly Printer
