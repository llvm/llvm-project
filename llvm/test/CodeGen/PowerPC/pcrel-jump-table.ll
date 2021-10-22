; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-R
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-use-absolute-jumptables \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-A-LE
; RUN: llc -verify-machineinstrs -target-abi=elfv2 -mtriple=powerpc64-- \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-R
; RUN: llc -verify-machineinstrs -target-abi=elfv2 -mtriple=powerpc64-- \
; RUN:   -mcpu=pwr10 -ppc-use-absolute-jumptables \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-A-BE


; This test checks for getting relative and absolute jump table base address
; using PC Relative addressing.

define dso_local signext i32 @jumptable(i32 signext %param) {
; CHECK-R-LABEL: jumptable:
; CHECK-R:       # %bb.1: # %entry
; CHECK-R-NEXT:    paddi r5, 0, .LJTI0_0@PCREL, 1
; CHECK-R-NEXT:    rldic r4, r4
; CHECK-R-NEXT:    lwax r4, r4, r5
; CHECK-R-NEXT:    add r4, r4, r5
; CHECK-R-NEXT:    mtctr r4
; CHECK-R-NEXT:    bctr
; CHECK-A-LE-LABEL: jumptable:
; CHECK-A-LE:       # %bb.1: # %entry
; CHECK-A-LE-NEXT:    rldic r4, r4
; CHECK-A-LE-NEXT:    paddi r5, 0, .LJTI0_0@PCREL, 1
; CHECK-A-LE-NEXT:    ldx r4, r4, r5
; CHECK-A-LE-NEXT:    mtctr r4
; CHECK-A-LE-NEXT:    bctr
; CHECK-A-BE-LABEL: jumptable:
; CHECK-A-BE:       # %bb.1: # %entry
; CHECK-A-BE-NEXT:    paddi r5, 0, .LJTI0_0@PCREL, 1
; CHECK-A-BE-NEXT:    rldic r4, r4
; CHECK-A-BE-NEXT:    lwax r4, r4, r5
; CHECK-A-BE-NEXT:    mtctr r4
; CHECK-A-BE-NEXT:    bctr


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
