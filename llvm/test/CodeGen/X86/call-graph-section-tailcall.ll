;; Tests that we store the type identifiers in .callgraph section of the object file for tailcalls.

; RUN: llc --call-graph-section -filetype=obj -o - < %s | \
; RUN: llvm-readelf -x .callgraph - | FileCheck %s

define dso_local noundef i32 @_Z13call_indirectPFicEc(ptr noundef readonly captures(none) %func, i8 noundef signext %x) local_unnamed_addr !type !0 {
entry:
  %call = tail call noundef i32 %func(i8 noundef signext %x), !callee_type !1
  ret i32 %call
}

define dso_local noundef i32 @main(i32 noundef %argc) local_unnamed_addr !type !3 {
entry:
  %0 = and i32 %argc, 1
  %cmp = icmp eq i32 %0, 0
  %_Z3fooc._Z3barc = select i1 %cmp, ptr @_Z3fooc, ptr @_Z3barc
  %call.i = tail call noundef i32 %_Z3fooc._Z3barc(i8 noundef signext 97), !callee_type !1
  ret i32 %call.i
}

declare !type !2 noundef i32 @_Z3fooc(i8 noundef signext) local_unnamed_addr

declare !type !2 noundef i32 @_Z3barc(i8 noundef signext) local_unnamed_addr

;; Check that the numeric type id (md5 hash) for the below type ids are emitted
;; to the callgraph section.

; CHECK: Hex dump of section '.callgraph':

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!2}
; CHECK-DAG: 5486bc59 814b8e30
!2 = !{i64 0, !"_ZTSFicE.generalized"}
!3 = !{i64 0, !"_ZTSFiiE.generalized"}
