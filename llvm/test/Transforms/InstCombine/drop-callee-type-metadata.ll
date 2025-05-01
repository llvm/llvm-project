;; Test if the callee_type metadata is dropped when it is attached
;; to a direct function call during instcombine.

; RUN: opt < %s -passes="instcombine" -disable-verify -S | FileCheck %s

define i32 @_Z3barv() local_unnamed_addr !type !3 {
entry:
  ; CHECK-LABEL: define i32 @_Z3barv()
  ; CHECK-NEXT: entry:
  ; CHECK-NOT: !callee_type
  ; CHECK-NEXT: %call = call i32 @_Z3fooc(i8 97)  
  %call = call i32 @_Z3fooc(i8 97), !callee_type !1
  ret i32 %call
}

declare !type !2 i32 @_Z3fooc(i8 signext)

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!2}
!2 = !{i64 0, !"_ZTSFicE.generalized"}
!3 = !{i64 0, !"_ZTSFivE.generalized"}
