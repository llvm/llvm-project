; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s

; Selection coverage in a mixed module. PASS3 must wrap every defined
; ejit_entry function, leave non-entry functions completely untouched, and
; skip ejit_entry-attributed declarations (no body to wrap).

; --- wrapped: static-only entry ---
; CHECK-LABEL: define void @entry_a()
; CHECK: jit_entry:
; CHECK: call ptr @ejit_compile_or_get
define void @entry_a() !ejit.metadata !0 {
entry:
  ret void
}

; --- wrapped: single-dim entry ---
; CHECK-LABEL: define void @entry_b(i8 %cell)
; CHECK: jit_entry:
; CHECK: call ptr @ejit_compile_or_get
define void @entry_b(i8 %cell) !ejit.metadata !1 {
entry:
  ret void
}

; --- untouched: ordinary function, no ejit metadata ---
; CHECK-LABEL: define void @plain_func()
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void
; CHECK-NOT: jit_entry
define void @plain_func() {
entry:
  ret void
}

; --- skipped: a bare declaration has no body and must be left as a
;     declaration. (LLVM IR cannot attach !ejit.metadata to a declaration, so
;     an "ejit_entry declaration" is unrepresentable; the pass's
;     !F.isDeclaration() guard covers any declaration regardless.) ---
; CHECK: declare void @extern_decl()
declare void @extern_decl()

!0 = distinct !{!{!"ejit_entry"}}
!1 = distinct !{!{!"ejit_entry"}, !{!"ejit_period_arr_ind", !"cell", i32 0}}
