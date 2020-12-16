; RUN: opt -enable-new-pm=0 -mtriple=x86_64-- -O0 -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK,%llvmcheckext

; REQUIRES: asserts

; Suppress FileCheck --allow-unused-prefixes=false diagnostics.
; CHECK-NOEXT: {{^}}

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Module Verifier
; CHECK-EXT:     Good Bye World Pass
; CHECK-NEXT:     Instrument function entry/exit with calls to e.g. mcount() (pre inlining)
; CHECK-NEXT: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Transform Information
;             Target Pass Configuration
; CHECK:      Assumption Cache Tracker
; CHECK-NEXT: Profile summary info
; CHECK-NEXT:   ModulePass Manager
; CHECK-NEXT:     Annotation2Metadata
; CHECK-NEXT:     Force set function attributes
; CHECK-NEXT:     CallGraph Construction
; CHECK-NEXT:     Call Graph SCC Pass Manager
; CHECK-NEXT:       Inliner for always_inline functions
;                   A No-Op Barrier Pass
; CHECK:            FunctionPass Manager
; CHECK-NEXT:         Annotation Remarks
; CHECK-NEXT:         Module Verifier
; CHECK-NEXT:     Bitcode Writer

define void @f() {
  ret void
}
