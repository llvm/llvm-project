; RUN: llc < %s -mtriple=thumb-apple-darwin -arm-global-merge -global-merge-group-by-use=false | FileCheck %s
; Test the GlobalMerge pass. Check that the pass does not crash when using
; multiple address spaces.

; CHECK: _MergedGlobals:
@g1 = internal addrspace(1) global i32 1
@g2 = internal addrspace(1) global i32 2


; CHECK: _MergedGlobals.1:
@g3 = internal addrspace(2) global i32 3
@g4 = internal addrspace(2) global i32 4
