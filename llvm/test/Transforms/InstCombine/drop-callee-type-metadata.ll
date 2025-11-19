;; Test if the callee_type metadata is dropped when it is attached
;; to a direct function call during instcombine.

; RUN: opt -passes=instcombine -S < %s | FileCheck %s

define i32 @_Z3barv() !type !0 {
; CHECK-LABEL: define i32 @_Z3barv(
; CHECK-SAME: ) !type [[META0:![0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @_Z3fooc(i8 97){{$}}
; CHECK-NEXT:    ret i32 [[CALL]]
;
entry:
  %call = call i32 @_Z3fooc(i8 97), !callee_type !1
  ret i32 %call
}

declare !type !2 i32 @_Z3fooc(i8 signext)

!0 = !{i64 0, !"_ZTSFivE.generalized"}
!1 = !{!2}
!2 = !{i64 0, !"_ZTSFicE.generalized"}
;.
; CHECK: [[META0]] = !{i64 0, !"_ZTSFivE.generalized"}
;.
