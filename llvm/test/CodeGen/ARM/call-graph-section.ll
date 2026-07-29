;; Tests that we store the type identifiers in .llvm.callgraph section of the object file.

; RUN: llc -mtriple=arm-unknown-linux --call-graph-section -filetype=obj -o - < %s | \
; RUN: llvm-readelf -x .llvm.callgraph - | FileCheck %s

declare !callgraph !0 void @foo()

declare !callgraph !1 i32 @bar(i8)

declare !callgraph !2 ptr @baz(ptr)

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
!0 = !{!"_ZTSFvE"}
!1 = !{!0}
!2 = !{!"_ZTSFicE"}
!3 = !{!2}
!4 = !{!5}
!5 = !{!"_ZTSFPvS_E"}

;; Make sure following type IDs are in call graph section
;; 0x5eecb3e2444f731f, 0x814b8e305486bc59, 0xf897fd777ade6814
; CHECK: Hex dump of section '.llvm.callgraph':
; CHECK-NEXT: 0x00000000 00050000 00000000 00000000 000003e4
; CHECK-NEXT: 0x00000010 2fa3e616 b06f5bc6 50144734 74f90140
; CHECK-NEXT: 0x00000020 f53f68ae 62b38c
