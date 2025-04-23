;; Tests that we store the type identifiers in .callgraph section of the object file.

; RUN: llc --call-graph-section -filetype=obj -o - < %s | \
; RUN: llvm-readelf -x .callgraph - | FileCheck %s

declare !type !0 void @foo()

declare !type !1 noundef i32 @bar(i8 signext)

declare !type !2 noundef ptr @baz(ptr)

define dso_local void @main() {
entry:
  %retval = alloca i32, align 4
  %fp_foo = alloca ptr, align 8
  %a = alloca i8, align 1
  %fp_bar = alloca ptr, align 8
  %fp_baz = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store ptr @foo, ptr %fp_foo, align 8
  %fp_foo_val = load ptr, ptr %fp_foo, align 8
  call void (...) %fp_foo_val(), !callee_type !1
  store ptr @bar, ptr %fp_bar, align 8
  %fp_bar_val = load ptr, ptr %fp_bar, align 8
  %a_val = load i8, ptr %a, align 1
  %call_fp_bar = call i32 %fp_bar_val(i8 signext %a_val), !callee_type !3
  store ptr @baz, ptr %fp_baz, align 8
  %fp_baz_val = load ptr, ptr %fp_baz, align 8
  %call_fp_baz = call ptr %fp_baz_val(ptr %a), !callee_type !5
  call void @foo()
  %a_val_2 = load i8, ptr %a, align 1
  %call_bar = call i32 @bar(i8 signext %a_val_2)
  %call_baz = call ptr @baz(ptr %a)
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
!4 = !{i64 0, !"_ZTSFPvS_E.generalized"}
!5 = !{!4}
