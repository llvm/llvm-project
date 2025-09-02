;; Tests that we store the type identifiers in .callgraph section of the object file.

; RUN: llc -mtriple=x86_64-unknown-linux --call-graph-section -filetype=obj -o - < %s | \
; RUN: llvm-readelf -x .callgraph - | FileCheck %s

declare !type !0 void @foo()

declare !type !1 i32 @bar(i8)

declare !type !2 ptr @baz(ptr)

define void @main() {
entry:
  %fp_foo_val = load ptr, ptr null, align 8
  call void (...) %fp_foo_val(), !callee_type !1
  %fp_bar_val = load ptr, ptr null, align 8
  %call_fp_bar = call i32 %fp_bar_val(i8 0), !callee_type !3
  %fp_baz_val = load ptr, ptr null, align 8
  %call_fp_baz = call ptr %fp_baz_val(ptr null), !callee_type !4
  ret void
}

;; Check that the numeric type id (md5 hash) for the below type ids are emitted
;; to the callgraph section.

; CHECK: Hex dump of section '.callgraph':

; CHECK-DAG: 2444f731 f5eecb3e
!0 = !{i64 0, !"_ZTSFvE.generalized"}
!1 = !{!0}
; CHECK-DAG: 5486bc59 814b8e30
!2 = !{i64 0, !"_ZTSFicE.generalized"}
!3 = !{!2}
; CHECK-DAG: 7ade6814 f897fd77
!4 = !{!5}
!5 = !{i64 0, !"_ZTSFPvS_E.generalized"}
