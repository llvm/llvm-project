; This test is mainly for the -outliner-leaf-descendants flag for MachineOutliner.
;
; ===================== -outliner-leaf-descendants=false =====================
; MachineOutliner finds THREE key `OutlinedFunction` and outlines them. They are:
;   ```
;     mov     w0, #1
;     mov     w1, #2
;     mov     w2, #3
;     mov     w3, #4
;     mov     w4, #5
;     mov     w5, #6 or #7 or #8
;     b
;   ```
; Each has:
;   - `SequenceSize=28` and `OccurrenceCount=2`
;   - each Candidate has `CallOverhead=4` and `FrameOverhead=0`
;   - `NotOutlinedCost=28*2=56` and `OutliningCost=4*2+28+0=36`
;   - `Benefit=56-36=20` and `Priority=56/36=1.56`
;
; ===================== -outliner-leaf-descendants=true =====================
; MachineOutliner finds a FOURTH key `OutlinedFunction`, which is:
;   ```
;   mov     w0, #1
;   mov     w1, #2
;   mov     w2, #3
;   mov     w3, #4
;   mov     w4, #5
;   ```
; This corresponds to an internal node that has ZERO leaf children, but SIX leaf descendants.
; It has:
;   - `SequenceSize=20` and `OccurrenceCount=6`
;   - each Candidate has `CallOverhead=12` and `FrameOverhead=4`
;   - `NotOutlinedCost=20*6=120` and `OutliningCost=12*6+20+4=96`
;   - `Benefit=120-96=24` and `Priority=120/96=1.25`
;
; The FOURTH `OutlinedFunction` has lower _priority_ compared to the first THREE `OutlinedFunction`.
; Hence, we use `-outliner-benefit-threshold=22` to check if the FOURTH `OutlinedFunction` is identified.

; RUN: llc %s -enable-machine-outliner=always -outliner-leaf-descendants=false -filetype=obj -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=CHECK-BASELINE

; RUN: llc %s -enable-machine-outliner=always -outliner-leaf-descendants=false -outliner-benefit-threshold=22 -filetype=obj -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=CHECK-NO-CANDIDATE

; RUN: llc %s -enable-machine-outliner=always -outliner-leaf-descendants=true -filetype=obj -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=CHECK-BASELINE

; RUN: llc %s -enable-machine-outliner=always -outliner-leaf-descendants=true -outliner-benefit-threshold=22 -filetype=obj -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=CHECK-LEAF-DESCENDANTS


target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

declare i32 @_Z3fooiiii(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef)

define i32 @_Z2f1v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6)
  ret i32 %1
}

define i32 @_Z2f2v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6)
  ret i32 %1
}

define i32 @_Z2f3v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 7)
  ret i32 %1
}

define i32 @_Z2f4v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 7)
  ret i32 %1
}

define i32 @_Z2f5v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 8)
  ret i32 %1
}

define i32 @_Z2f6v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 8)
  ret i32 %1
}

; CHECK-BASELINE: <_OUTLINED_FUNCTION_0>:
; CHECK-BASELINE-NEXT: mov     w0, #0x1
; CHECK-BASELINE-NEXT: mov     w1, #0x2
; CHECK-BASELINE-NEXT: mov     w2, #0x3
; CHECK-BASELINE-NEXT: mov     w3, #0x4
; CHECK-BASELINE-NEXT: mov     w4, #0x5
; CHECK-BASELINE-NEXT: mov     w5, #0x6
; CHECK-BASELINE-NEXT: b

; CHECK-BASELINE: <_OUTLINED_FUNCTION_1>:
; CHECK-BASELINE-NEXT: mov     w0, #0x1
; CHECK-BASELINE-NEXT: mov     w1, #0x2
; CHECK-BASELINE-NEXT: mov     w2, #0x3
; CHECK-BASELINE-NEXT: mov     w3, #0x4
; CHECK-BASELINE-NEXT: mov     w4, #0x5
; CHECK-BASELINE-NEXT: mov     w5, #0x8
; CHECK-BASELINE-NEXT: b

; CHECK-BASELINE: <_OUTLINED_FUNCTION_2>:
; CHECK-BASELINE-NEXT: mov     w0, #0x1
; CHECK-BASELINE-NEXT: mov     w1, #0x2
; CHECK-BASELINE-NEXT: mov     w2, #0x3
; CHECK-BASELINE-NEXT: mov     w3, #0x4
; CHECK-BASELINE-NEXT: mov     w4, #0x5
; CHECK-BASELINE-NEXT: mov     w5, #0x7
; CHECK-BASELINE-NEXT: b

; CHECK-LEAF-DESCENDANTS: <_OUTLINED_FUNCTION_0>:
; CHECK-LEAF-DESCENDANTS-NEXT: mov     w0, #0x1
; CHECK-LEAF-DESCENDANTS-NEXT: mov     w1, #0x2
; CHECK-LEAF-DESCENDANTS-NEXT: mov     w2, #0x3
; CHECK-LEAF-DESCENDANTS-NEXT: mov     w3, #0x4
; CHECK-LEAF-DESCENDANTS-NEXT: mov     w4, #0x5
; CHECK-LEAF-DESCENDANTS-NEXT: ret

; CHECK-LEAF-DESCENDANTS-NOT: <_OUTLINED_FUNCTION_1>:

; CHECK-NO-CANDIDATE-NOT: <_OUTLINED_FUNCTION_0>:
