; RUN: llc < %s -mtriple=arm-eabi -mattr=+v4t -O0 | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK_O0
; RUN: llc < %s -mtriple=arm-eabi -mattr=+v4t -O1 | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK_O1

; In /O0, the addition must not be eliminated. This happens when the load
; and store are folded by the DAGCombiner. In /O1 and above, the optimization
; must be executed.

; CHECK-LABEL:   {{^}}main
; CHECK:         mov [[TMP:r[0-9]+]], #0
; CHECK-NEXT:    str [[TMP]], [sp, #4]
; CHECK_O0:      str [[TMP]], [sp]
; CHECK_O0:      ldr [[TMP:r[0-9]+]], [sp]
; CHECK_O0-NEXT: add [[TMP]], [[TMP]], #2
; CHECK_O1-NOT:  ldr [[TMP:r[0-9]+]], [sp]
; CHECK_O1-NOT:  add [[TMP]], [[TMP]], #2

define i32 @main() {
bb:
  %tmp = alloca i32, align 4
  %tmp1 = alloca i32, align 4
  store i32 0, i32* %tmp, align 4
  store i32 0, i32* %tmp1, align 4
  %tmp2 = load i32, i32* %tmp1, align 4
  %tmp3 = add nsw i32 %tmp2, 2
  store i32 %tmp3, i32* %tmp1, align 4
  %tmp4 = load i32, i32* %tmp1, align 4
  %tmp5 = icmp eq i32 %tmp4, 2
  br i1 %tmp5, label %bb6, label %bb7

bb6:                                              ; preds = %bb
  store i32 0, i32* %tmp, align 4
  br label %bb8

bb7:                                              ; preds = %bb
  store i32 5, i32* %tmp, align 4
  br label %bb8

bb8:                                              ; preds = %bb7, %bb6
  %tmp9 = load i32, i32* %tmp, align 4
  ret i32 %tmp9
}
