; RUN: opt -S -passes="ipsccp<func-spec>" -force-specialization < %s | FileCheck %s

;; FunctionSpecialization clones a function with a function-pointer constant
;; substituted into the body. IPSCCP then rewrites the indirect call
;;
;;   call void %fn_param() [ "ptrauth"(i32 0, i64 0) ]   ; LEGAL indirect
;;
;; into a direct call by replacing %fn_param with the Function constant. The
;; operand bundle is not touched by replaceAllUsesWith, so the result would be
;;
;;   call void @callee() [ "ptrauth"(i32 0, i64 0) ]     ; ILLEGAL direct
;;
;; which the verifier rejects ("Direct call cannot have a ptrauth bundle").
;; The fix in FunctionSpecializer::createSpecialization strips ptrauth bundles
;; from calls in the clone whose callee is one of the formal arguments being
;; specialized to a Function constant, before IPSCCP propagates the constant.
;;
;; Callers pass *different* function constants so IPSCCP cannot do in-place
;; propagation and must rely on FunctionSpecialization to clone the helper
;; per constant — which is the path that triggers the original Kotlin/Native
;; arm64e crash.

target triple = "arm64e-apple-ios14.0.0"

@global_fn_ptr = external global ptr

define void @callee1() {
entry:
  ret void
}

define void @callee2() {
entry:
  ret void
}

;; ---------------------------------------------------------------------------
;; Positive (call form): bundle MUST be stripped from the devirtualized call.
;; ---------------------------------------------------------------------------

define internal void @helper_call(ptr %fn) {
entry:
  call void %fn() [ "ptrauth"(i32 0, i64 0) ]
  ret void
}

define void @caller_call_a() {
  call void @helper_call(ptr @callee1)
  ret void
}

define void @caller_call_b() {
  call void @helper_call(ptr @callee2)
  ret void
}

;; FuncSpec's global .specialized.N counter is fragile to traversal order, so
;; we only require the clones exist — the verifier rejects ill-formed direct
;; calls, so the mere fact that opt produces output proves no devirtualized
;; call carries a ptrauth bundle.
; CHECK-DAG: @helper_call.specialized
; CHECK-DAG: @helper_call.specialized

;; ---------------------------------------------------------------------------
;; Positive (invoke form): bundle MUST be stripped from the devirtualized
;; invoke. This mirrors the original Kotlin/Native crash shape.
;; ---------------------------------------------------------------------------

declare i32 @__gxx_personality_v0(...)

define internal void @helper_invoke(ptr %fn) personality ptr @__gxx_personality_v0 {
entry:
  invoke void %fn() [ "ptrauth"(i32 0, i64 0) ]
          to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}

define void @caller_invoke_a() {
  call void @helper_invoke(ptr @callee1)
  ret void
}

define void @caller_invoke_b() {
  call void @helper_invoke(ptr @callee2)
  ret void
}

; CHECK-DAG: @helper_invoke.specialized
; CHECK-DAG: @helper_invoke.specialized

;; ---------------------------------------------------------------------------
;; Negative: a ptrauth bundle on a still-indirect call (callee is loaded from
;; a global, not from the substituted argument) MUST be preserved. Guards
;; against an over-eager fix that strips bundles unconditionally.
;; ---------------------------------------------------------------------------

define internal void @helper_mixed(ptr %fn) {
entry:
  ; Devirtualized — bundle stripped.
  call void %fn() [ "ptrauth"(i32 0, i64 0) ]
  ; Still indirect — bundle preserved.
  %loaded = load ptr, ptr @global_fn_ptr
  call void %loaded() [ "ptrauth"(i32 0, i64 0) ]
  ret void
}

define void @caller_mixed_a() {
  call void @helper_mixed(ptr @callee1)
  ret void
}

define void @caller_mixed_b() {
  call void @helper_mixed(ptr @callee2)
  ret void
}

; CHECK-DAG: @helper_mixed.specialized
; CHECK-DAG: @helper_mixed.specialized

;; The mixed helper's still-indirect call (through a loaded fn-pointer)
;; retains its ptrauth bundle in every clone. Guards against an over-eager
;; fix that strips bundles unconditionally.
; CHECK-COUNT-2: call void %{{.*}}() [ "ptrauth"(i32 0, i64 0) ]
