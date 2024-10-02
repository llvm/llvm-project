; RUN: llc -mtriple=dxil-pc-shadermodel6.3-library -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN:     grep -v "Verify generated machine code" | FileCheck %s

; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: ModulePass Manager
; CHECK-NEXT:   DXIL Intrinsic Expansion
; CHECK-NEXT:   DXIL Data Scalarization
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     Scalarize vector operations
; CHECK-NEXT:   DXIL Intrinsic Expansion
; CHECK-NEXT:   DXIL Resource analysis
; CHECK-NEXT:   DXIL Op Lowering
; CHECK-NEXT:   DXIL Finalize Linkage
; CHECK-NEXT:   DXIL resource Information
; CHECK-NEXT:   DXIL Shader Flag Analysis
; CHECK-NEXT:   DXIL Module Metadata analysis
; CHECK-NEXT:   DXIL Translate Metadata
; CHECK-NEXT:   DXIL Prepare Module
; CHECK-NEXT:   DXIL Metadata Pretty Printer
; CHECK-NEXT:   Print Module IR
 
