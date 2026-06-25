; RUN: opt -passes=ejit-wrapper-gen -disable-output %s
;
; XFAIL: *
;
; KNOWN BUG (recorded now, fix tracked separately).
;
; EJitWrapperGen unconditionally adds the `noinline` attribute to every
; ejit_entry function (EJitNoInlineEntry defaults to true). If the user wrote
;     __attribute__((always_inline)) ejit_entry ...
; the function already carries `alwaysinline`, and `noinline` + `alwaysinline`
; are mutually exclusive — the module verifier aborts the whole AOT compile:
;     Attributes 'noinline and alwaysinline' are incompatible!
;     LLVM ERROR: Broken module found, compilation aborted!
;
; This RUN line currently aborts (non-zero exit) -> the test fails -> XFAIL
; matches. The fix should make the pass not create conflicting attributes
; (e.g. skip adding noinline when alwaysinline is present, or drop
; alwaysinline). Once fixed, opt's built-in verifier passes, this RUN
; succeeds, and lit reports XPASS -> remove the XFAIL line.

define void @always_inline_entry() alwaysinline !ejit.metadata !0 {
entry:
  ret void
}

!0 = distinct !{!{!"ejit_entry"}}
