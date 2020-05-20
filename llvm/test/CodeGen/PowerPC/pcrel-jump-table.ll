; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-R
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-use-absolute-jumptables \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-A

; This test checks for getting relative and absolute jump table base address
; using PC Relative addressing.

define dso_local signext i32 @jumptable(i32 signext %param) {
; CHECK-R-LABEL: jumptable:
; CHECK-R:       # %bb.1: # %entry
; CHECK-R-NEXT:    rldic r4, r4
; CHECK-R-NEXT:    paddi r5, 0, .LJTI0_0@PCREL, 1
; CHECK-R-NEXT:    lwax r4, r4, r5
; CHECK-R-NEXT:    add r4, r4, r5
; CHECK-R-NEXT:    mtctr r4
; CHECK-R-NEXT:    bctr
; CHECK-A-LABEL: jumptable:
; CHECK-A:       # %bb.1: # %entry
; CHECK-A-NEXT:    rldic r4, r4
; CHECK-A-NEXT:    paddi r5, 0, .LJTI0_0@PCREL, 1
; CHECK-A-NEXT:    ldx r4, r4, r5
; CHECK-A-NEXT:    mtctr r4
; CHECK-A-NEXT:    bctr

entry:
  switch i32 %param, label %sw.default [
    i32 1, label %return
    i32 2, label %sw.bb1
    i32 3, label %sw.bb2
    i32 4, label %sw.bb3
    i32 20, label %sw.bb4
  ]

sw.bb1:                                           ; preds = %entry
  br label %return

sw.bb2:                                           ; preds = %entry
  br label %return

sw.bb3:                                           ; preds = %entry
  br label %return

sw.bb4:                                           ; preds = %entry
  br label %return

sw.default:                                       ; preds = %entry
  br label %return

return:  ; preds = %entry, %sw.default, %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1
  %retval.0 = phi i32 [ -1, %sw.default ], [ 400, %sw.bb4 ], [ 16, %sw.bb3 ],
                      [ 9, %sw.bb2 ], [ 4, %sw.bb1 ], [ %param, %entry ]
  ret i32 %retval.0
}
