; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s
;
; XFAIL: *
;
; KNOWN BUG (recorded now, fix tracked separately).
;
; A variadic ejit_entry function is wrapped like any other, but the jit_dispatch
; block builds its argument list only from F->args() — the NAMED parameters.
; The variadic ("...") arguments are silently dropped from the indirect call:
;     %r = call i32 (i32, ...) %pfn(i32 %n)      ; the ... args are gone
; so the JIT-specialized path receives garbage for everything past %n.
;
; The variadic tail cannot be re-forwarded in IR without musttail, and musttail
; is impossible here because the dispatch call is not in tail position (a
; fallback path follows). Therefore the correct fix is to NOT wrap variadic
; functions at all — leave the function untouched so it always runs AOT.
;
; The CHECK below encodes that fix (no wrapper prologue is generated). Today the
; function IS wrapped, so the CHECK-NOT fails -> XFAIL.

define i32 @vararg_entry(i32 %n, ...) !ejit.metadata !0 {
entry:
  ret i32 %n
}

!0 = distinct !{!{!"ejit_entry"}}

; A variadic ejit_entry must be left as-is (no JIT wrapper inserted).
; CHECK-LABEL: define i32 @vararg_entry(i32 %n, ...)
; CHECK-NOT: jit_entry:
; CHECK-NOT: ejit_compile_or_get
