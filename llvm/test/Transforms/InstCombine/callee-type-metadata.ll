;; Test if the callee_type metadata is dropped when an indirect function call through a function ptr is promoted
;; to a direct function call during instcombine.

; RUN: opt < %s -O2 | llvm-dis | FileCheck %s

define dso_local noundef i32 @_Z13call_indirectPFicEc(ptr noundef %func, i8 noundef signext %x) local_unnamed_addr !type !0 {
entry:
  %call = call noundef i32 %func(i8 noundef signext %x), !callee_type !1
  ret i32 %call
}

define dso_local noundef i32 @_Z3barv() local_unnamed_addr !type !3 {
entry:
  ; CHECK: %call.i = tail call noundef i32 @_Z3fooc(i8 noundef signext 97)
  ; CHECK-NOT: %call.i = tail call noundef i32 @_Z3fooc(i8 noundef signext 97), !callee_type !1
  %call = call noundef i32 @_Z13call_indirectPFicEc(ptr noundef nonnull @_Z3fooc, i8 noundef signext 97)
  ret i32 %call
}

declare !type !2 noundef i32 @_Z3fooc(i8 noundef signext)

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!2}
!2 = !{i64 0, !"_ZTSFicE.generalized"}
!3 = !{i64 0, !"_ZTSFivE.generalized"}
