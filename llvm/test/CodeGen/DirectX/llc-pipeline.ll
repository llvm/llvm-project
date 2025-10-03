; RUN: llc -filetype=asm -mtriple=dxil-pc-shadermodel6.3-library -debug-pass=Structure < %s -o /dev/null 2>&1 | grep -v "Verify generated machine code" | FileCheck %s --check-prefixes=CHECK,CHECK-ASM
; RUN: llc -filetype=obj -mtriple=dxil-pc-shadermodel6.3-library -debug-pass=Structure < %s -o /dev/null 2>&1 | grep -v "Verify generated machine code" | FileCheck %s --check-prefixes=CHECK,CHECK-OBJ

; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: DXIL Resource Type Analysis
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-OBJ-NEXT: Machine Module Information
; CHECK-OBJ-NEXT: Machine Branch Probability Analysis
; CHECK-OBJ-NEXT: Create Garbage Collector Module Metadata

; CHECK-NEXT: ModulePass Manager
; CHECK-NEXT:   DXIL Finalize Linkage
; CHECK-NEXT:   Dead Global Elimination
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     DXIL Resource Access
; CHECK-NEXT:   DXIL Intrinsic Expansion
; CHECK-NEXT:   DXIL CBuffer Access
; CHECK-NEXT:   DXIL Data Scalarization
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     Scalarize vector operations
; CHECK-NEXT:   DXIL Array Flattener
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     DXIL Forward Handle Accesses
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:     Function Alias Analysis Results
; CHECK-NEXT:     Post-Dominator Tree Construction
; CHECK-NEXT:     Memory SSA
; CHECK-NEXT:     Natural Loop Information
; CHECK-NEXT:     Dead Store Elimination
; CHECK-NEXT:     DXIL Legalizer
; CHECK-NEXT:   DXIL Resource Binding Analysis
; CHECK-NEXT:   DXIL Resource Implicit Binding
; CHECK-NEXT:   DXIL Resources Analysis
; CHECK-NEXT:   DXIL Module Metadata analysis
; CHECK-NEXT:   DXIL Shader Flag Analysis
; CHECK-NEXT:   DXIL Translate Metadata
; CHECK-NEXT:   DXIL Root Signature Analysis
; CHECK-NEXT:   DXIL Post Optimization Validation
; CHECK-NEXT:   DXIL Op Lowering
; CHECK-NEXT:   DXIL Prepare Module

; CHECK-ASM-NEXT: DXIL Metadata Pretty Printer
; CHECK-ASM-NEXT: Print Module IR

; CHECK-OBJ-NEXT: DXIL Embedder
; CHECK-OBJ-NEXT: DXContainer Global Emitter
; CHECK-OBJ-NEXT: FunctionPass Manager
; CHECK-OBJ-NEXT:   Lazy Machine Block Frequency Analysis
; CHECK-OBJ-NEXT:   Machine Optimization Remark Emitter
; CHECK-OBJ-NEXT:   DXIL Assembly Printer
