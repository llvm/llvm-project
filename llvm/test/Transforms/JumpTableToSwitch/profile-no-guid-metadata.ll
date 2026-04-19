; RUN: opt < %s -passes=jump-table-to-switch -S | FileCheck %s

;; Test that when target functions lack !guid metadata, the pass correctly
;; computes GUIDs from the target function names (not the caller's name)
;; for matching against value profile data. This is a regression test for
;; a bug where the caller function was used instead of the callee when
;; computing PGO function names for GUID lookup.

@jt = constant [2 x ptr] [ptr @jt_target_0, ptr @jt_target_1]

;; Note: these functions intentionally do NOT have !guid metadata,
;; forcing the pass to compute GUIDs via getIRPGOFuncName.
define i32 @jt_target_0() {
  ret i32 10
}

define i32 @jt_target_1() {
  ret i32 20
}

define i32 @caller(i32 %idx) {
; CHECK-LABEL: define i32 @caller(
; CHECK:         switch i32 [[IDX:%.*]], label %{{.*}} [
; CHECK-NEXT:      i32 0, label %[[CALL0:.*]]
; CHECK-NEXT:      i32 1, label %[[CALL1:.*]]
; CHECK-NEXT:    ], !prof [[PROF:![0-9]+]]
; CHECK:       [[CALL0]]:
; CHECK-NEXT:    {{.*}} = call i32 @jt_target_0()
; CHECK:       [[CALL1]]:
; CHECK-NEXT:    {{.*}} = call i32 @jt_target_1()
  %gep = getelementptr inbounds [2 x ptr], ptr @jt, i32 0, i32 %idx
  %fptr = load ptr, ptr %gep
  %r = call i32 %fptr(), !prof !0
  ret i32 %r
}

;; VP metadata: GUID 11912887233601027218 = MD5("jt_target_0"), count 100
;;              GUID 18156790114353049777 = MD5("jt_target_1"), count 50
!0 = !{!"VP", i32 0, i64 150, i64 11912887233601027218, i64 100, i64 18156790114353049777, i64 50}

; CHECK: [[PROF]] = !{!"branch_weights", i32 0, i32 100, i32 50}
