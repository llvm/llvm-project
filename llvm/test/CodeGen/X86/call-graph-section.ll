;; Tests that we store the type identifiers in .llvm.callgraph section of the object file.

; REQUIRES: x86-registered-target
; REQUIRES: arm-registered-target

; RUN: llc -mtriple=x86_64-unknown-linux --call-graph-section -filetype=obj -o - < %s | \
; RUN: llvm-readelf -x .llvm.callgraph - | FileCheck %s

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
!0 = !{i64 0, !"_ZTSFvE.generalized"}
!1 = !{!0}
!2 = !{i64 0, !"_ZTSFicE.generalized"}
!3 = !{!2}
!4 = !{!5}
!5 = !{i64 0, !"_ZTSFPvS_E.generalized"}

;; Make sure following type IDs are in call graph section
;; 0x5eecb3e2444f731f, 0x814b8e305486bc59, 0xf897fd777ade6814
; CHECK:      Hex dump of section '.llvm.callgraph':
; CHECK-NEXT: 0x00000000 00050000 00000000 00000000 00000000
; CHECK-NEXT: 0x00000010 00000324 44f731f5 eecb3e54 86bc5981
; CHECK-NEXT: 0x00000020 4b8e307a de6814f8 97fd77
