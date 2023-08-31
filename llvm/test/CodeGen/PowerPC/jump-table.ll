; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -stop-after=finalize-isel \
; RUN: | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-ibm-aix -stop-after=finalize-isel \
; RUN: | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-ibm-aix -stop-after=finalize-isel \
; RUN: | FileCheck %s

; CHECK:   jumpTable:
; CHECK-NEXT:   kind:            label-difference32
; CHECK-NEXT:   entries:
; CHECK-NEXT:     - id:              0

define signext i32 @jt(i32 signext %a, i32 signext %b) {
entry:
  switch i32 %a, label %sw.epilog [
    i32 15, label %return
    i32 12, label %sw.bb1
    i32 19, label %sw.bb2
    i32 27, label %sw.bb3
    i32 31, label %sw.bb4
  ]

sw.bb1:                                           ; preds = %entry
  br label %return

sw.bb2:                                           ; preds = %entry
  br label %return

sw.bb3:                                           ; preds = %entry
  br label %return

sw.bb4:                                           ; preds = %entry
  br label %return

sw.epilog:                                        ; preds = %entry
  br label %return

return:                                           ; preds = %entry, %sw.epilog, %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1
  %retval.0 = phi i32 [ 0, %sw.epilog ], [ 51, %sw.bb4 ], [ 49, %sw.bb3 ], [ 48, %sw.bb2 ], [ 46, %sw.bb1 ], [ 45, %entry ]
  ret i32 %retval.0
}
