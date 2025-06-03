; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s < %t

; Make sure there's no assertion for invoke destinations that don't
; use landingpad (and use catchswitch instead)

; CHECK-INTERESTINGNESS: invoke

; CHECK-FINAL: bb:
; CHECK-FINAL-NEXT: invoke void @llvm.seh.try.begin()
; CHECK-FINAL-NEXT:   to label %bb7 unwind label %bb1
; CHECK-FINAL: bb1:
; CHECK-FINAL-NEXT: %i = catchswitch within none [label %bb2] unwind to caller

; CHECK-FINAL: bb2:
; CHECK-FINAL-NEXT: %i3 = catchpad within %i [ptr null]
; CHECK-FINAL-NEXT: ret ptr null

; CHECK-FINAL-NOT: bb4
; CHECK-FINAL-NOT: bb5

; CHECK-FINAL: bb7:
; CHECK-FINAL-NEXT: ret ptr null
define ptr @func() personality ptr @__C_specific_handler {
bb:
  invoke void @llvm.seh.try.begin()
          to label %bb7 unwind label %bb1

bb1:                                              ; preds = %bb
  %i = catchswitch within none [label %bb2] unwind to caller

bb2:                                              ; preds = %bb1
  %i3 = catchpad within %i [ptr null]
  catchret from %i3 to label %bb4

bb4:                                              ; preds = %bb2
  invoke void @llvm.seh.try.end()
          to label %bb7 unwind label %bb5

bb5:                                              ; preds = %bb4
  %i6 = cleanuppad within none []
  cleanupret from %i6 unwind to caller

bb7:                                              ; preds = %bb4, %bb
  ret ptr null
}

declare void @llvm.seh.try.begin() #0
declare void @llvm.seh.try.end() #0
declare i32 @__C_specific_handler(...)

attributes #0 = { nounwind willreturn memory(write) }

