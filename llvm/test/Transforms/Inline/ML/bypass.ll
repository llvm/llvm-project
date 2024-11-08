; REQUIRES: have_tflite
; RUN: rm -rf %t.runfiles %t.tflite %t.model_out
; RUN: mkdir %t.runfiles
; RUN: cp %S/../../../../lib/Analysis/models/gen-inline-oz-test-model.py %t.runfiles
; RUN: cp %S/../../../../lib/Analysis/models/saved-model-to-tflite.py %t.runfiles
; RUN: %python %t.runfiles/gen-inline-oz-test-model.py %t.model_out never
; RUN: %python %t.runfiles/saved-model-to-tflite.py %t.model_out %t.tflite

; When running O2, we expect both callers to inline callee.
; RUN: opt < %s -passes='default<O2>' -inline-threshold=0 -hot-callsite-threshold=100 -S | FileCheck %s --check-prefixes=O2-HOT,O2-COLD

; The ML model we use always blocks inlining (by construction)
; RUN: opt < %s -passes='default<O2>' -inline-threshold=0 -hot-callsite-threshold=100 \
; RUN:  -enable-ml-inliner=development -ml-inliner-model-under-training=%t.tflite \
; RUN:  -S | FileCheck %s --check-prefixes=ML-HOT,ML-COLD

; When bypassing ML for non-cold callers, the hot caller will have its callee inlined, but the cold one won't
; RUN: opt < %s -passes='default<O2>' -inline-threshold=0 -hot-callsite-threshold=100 \
; RUN:  -enable-ml-inliner=development -ml-inliner-model-under-training=%t.tflite \
; RUN: -ml-inliner-skip-policy=if-caller-not-cold -S | FileCheck %s --check-prefixes=O2-HOT,ML-COLD

declare void @extern()

define i32 @callee(i32 %x) {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  call void @extern()
  call void @extern()
  ret i32 %x3
}

define i32 @hot_caller(i32 %y1) !prof !15 {
  %y = call i32 @callee(i32 %y1), !prof !16
  ret i32 %y
}

define i32 @cold_caller(i32 %y1) !prof !17 {
  %y = call i32 @callee(i32 %y1), !prof !16
  ret i32 %y
}


!llvm.module.flags = !{!1}
!15 = !{!"function_entry_count", i64 300}
!16 = !{!"branch_weights", i64 300}
!17 = !{!"function_entry_count", i64 1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"SampleProfile"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 1000}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}

; O2-HOT-LABEL: @hot_caller
; O2-HOT-NOT: call i32 @callee
; O2-HOT: call void @extern
; O2-HOT-NEXT: call void @extern
; O2-HOT-NEXT: ret
; O2-COLD-LABEL: @cold_caller
; O2-COLD-NOT: call i32 @callee
; O2-COLD: call void @extern
; O2-COLD-NEXT: call void @extern
; O2-COLD-NEXT: ret

; ML-HOT-LABEL: @hot_caller
; ML-HOT-NEXT: call i32 @callee
; ML-COLD-LABEL: @cold_caller
; ML-COLD-NEXT: call i32 @callee