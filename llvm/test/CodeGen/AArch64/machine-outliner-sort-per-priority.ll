; This tests the order in which functions are outlined in MachineOutliner
; There are TWO key OutlinedFunction in FunctionList
;
; ===================== First One =====================
;   ```
;     mov     w0, #1
;     mov     w1, #2
;     mov     w2, #3
;     mov     w3, #4
;     mov     w4, #5
;   ```
; It has:
;   - `SequenceSize=20` and `OccurrenceCount=6`
;   - each Candidate has `CallOverhead=12` and `FrameOverhead=4`
;   - `NotOutlinedCost=20*6=120` and `OutliningCost=12*6+20+4=96`
;   - `Benefit=120-96=24` and `Priority=120/96=1.25`
;
; ===================== Second One =====================
;   ```
;     mov     w6, #6
;     mov     w7, #7
;     b
;   ```
; It has:
;   - `SequenceSize=12` and `OccurrenceCount=4`
;   - each Candidate has `CallOverhead=4` and `FrameOverhead=0`
;   - `NotOutlinedCost=12*4=48` and `OutliningCost=4*4+12+0=28`
;   - `Benefit=48-28=20` and `Priority=48/28=1.71`
;
; Note that the first one has higher benefit, but lower priority.
; Hence, when outlining per priority, the second one will be outlined first.

; RUN: llc %s -enable-machine-outliner=always -filetype=obj -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=CHECK-SORT-BY-PRIORITY

; RUN: llc %s -enable-machine-outliner=always -outliner-benefit-threshold=22 -filetype=obj -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=CHECK-THRESHOLD


target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

declare i32 @_Z3fooiiii(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef)

define i32 @_Z2f1v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 11, i32 noundef 6, i32 noundef 7)
  ret i32 %1
}

define i32 @_Z2f2v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 12, i32 noundef 6, i32 noundef 7)
  ret i32 %1
}

define i32 @_Z2f3v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 13, i32 noundef 6, i32 noundef 7)
  ret i32 %1
}

define i32 @_Z2f4v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 14, i32 noundef 6, i32 noundef 7)
  ret i32 %1
}

define i32 @_Z2f5v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 15, i32 noundef 8, i32 noundef 9)
  ret i32 %1
}

define i32 @_Z2f6v() minsize {
  %1 = tail call i32 @_Z3fooiiii(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 16, i32 noundef 9, i32 noundef 8)
  ret i32 %1
}

; CHECK-SORT-BY-PRIORITY: <_OUTLINED_FUNCTION_0>:
; CHECK-SORT-BY-PRIORITY-NEXT: mov     w6, #0x6
; CHECK-SORT-BY-PRIORITY-NEXT: mov     w7, #0x7
; CHECK-SORT-BY-PRIORITY-NEXT: b

; CHECK-SORT-BY-PRIORITY: <_OUTLINED_FUNCTION_1>:
; CHECK-SORT-BY-PRIORITY-NEXT: mov     w0, #0x1
; CHECK-SORT-BY-PRIORITY-NEXT: mov     w1, #0x2
; CHECK-SORT-BY-PRIORITY-NEXT: mov     w2, #0x3
; CHECK-SORT-BY-PRIORITY-NEXT: mov     w3, #0x4
; CHECK-SORT-BY-PRIORITY-NEXT: mov     w4, #0x5
; CHECK-SORT-BY-PRIORITY-NEXT: ret

; CHECK-THRESHOLD: <_OUTLINED_FUNCTION_0>:
; CHECK-THRESHOLD-NEXT: mov     w0, #0x1
; CHECK-THRESHOLD-NEXT: mov     w1, #0x2
; CHECK-THRESHOLD-NEXT: mov     w2, #0x3
; CHECK-THRESHOLD-NEXT: mov     w3, #0x4
; CHECK-THRESHOLD-NEXT: mov     w4, #0x5
; CHECK-THRESHOLD-NEXT: ret

; CHECK-THRESHOLD-NOT: <_OUTLINED_FUNCTION_1>:
