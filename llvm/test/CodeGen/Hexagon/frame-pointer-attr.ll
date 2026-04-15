; RUN: llc -mtriple=hexagon -O2 -o - %s | FileCheck %s

; The "frame-pointer"="all" attribute (i.e. -fno-omit-frame-pointer) must be
; respected even for leaf functions that have no stack usage. Hexagon's
; hasFPImpl was incorrectly gating the DisableFramePointerElim check behind
; MFI.getStackSize() > 0, so leaf functions with no stack would skip the
; check and omit the frame pointer.

; CHECK-LABEL: leaf_fp:
; CHECK: allocframe
define i32 @leaf_fp(i32 %a, i32 %b) #0 {
entry:
  %add = add nsw i32 %a, %b
  ret i32 %add
}

; Verify that without the attribute, the leaf function still omits the
; frame pointer (the optimization is preserved).
; CHECK-LABEL: leaf_no_fp:
; CHECK-NOT: allocframe
define i32 @leaf_no_fp(i32 %a, i32 %b) #1 {
entry:
  %add = add nsw i32 %a, %b
  ret i32 %add
}

attributes #0 = { nounwind "frame-pointer"="all" }
attributes #1 = { nounwind }
