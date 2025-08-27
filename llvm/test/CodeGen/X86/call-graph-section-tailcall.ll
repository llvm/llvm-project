;; Tests that we store the type identifiers in .callgraph section of the object file for tailcalls.

; RUN: llc -mtriple=x86_64-unknown-linux --call-graph-section -filetype=obj -o - < %s | \
; RUN: llvm-readelf -x .callgraph - | FileCheck %s

define i32 @check_tailcall(ptr %func, i8 %x) !type !0 {
entry:
  %call = tail call i32 %func(i8 signext %x), !callee_type !1
  ret i32 %call
}

define i32 @main(i32 %argc) !type !3 {
entry:
  %andop = and i32 %argc, 1
  %cmp = icmp eq i32 %andop, 0
  %foo.bar = select i1 %cmp, ptr @foo, ptr @bar
  %call.i = tail call i32 %foo.bar(i8 signext 97), !callee_type !1
  ret i32 %call.i
}

declare !type !2 i32 @foo(i8 signext)

declare !type !2 i32 @bar(i8 signext)

;; Check that the numeric type id (md5 hash) for the below type ids are emitted
;; to the callgraph section.

; CHECK: Hex dump of section '.callgraph':

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!2}
; CHECK-DAG: 5486bc59 814b8e30
!2 = !{i64 0, !"_ZTSFicE.generalized"}
!3 = !{i64 0, !"_ZTSFiiE.generalized"}
