; RUN: opt -S -passes="ipsccp<func-spec>" -force-specialization < %s | FileCheck %s

;; After FunctionSpecialization clones a helper and IPSCCP RAUWs the called
;; operand with a Function constant, the original ptrauth bundle remains on
;; the now-direct call. The verifier must accept this and opt must not abort.
;; Distinct constants per caller force the FuncSpec clone path.

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

;; Call form: devirtualized direct call retains its (no-op) bundle.

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

;; .specialized.N suffix is traversal-order-dependent, so match any N. The
;; CHECK-DAG lines in this file are one group and their matches cannot overlap,
;; so each repeated line needs its own clone.
; CHECK-DAG: define internal void @helper_call.specialized.{{[0-9]+}}(
; CHECK-DAG: define internal void @helper_call.specialized.{{[0-9]+}}(
; CHECK-DAG: call void @callee1() [ "ptrauth"(i32 0, i64 0) ]
; CHECK-DAG: call void @callee2() [ "ptrauth"(i32 0, i64 0) ]

;; Invoke form: same shape with an unwind edge.

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

; CHECK-DAG: define internal void @helper_invoke.specialized.{{[0-9]+}}(
; CHECK-DAG: define internal void @helper_invoke.specialized.{{[0-9]+}}(
; CHECK-DAG: invoke void @callee1() [ "ptrauth"(i32 0, i64 0) ]
; CHECK-DAG: invoke void @callee2() [ "ptrauth"(i32 0, i64 0) ]

;; Mixed form: bundle on a still-indirect call must be preserved.

define internal void @helper_mixed(ptr %fn) {
entry:
  call void %fn() [ "ptrauth"(i32 0, i64 0) ]
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

; CHECK-DAG: define internal void @helper_mixed.specialized.{{[0-9]+}}(
; CHECK-DAG: define internal void @helper_mixed.specialized.{{[0-9]+}}(
; CHECK-DAG: call void @callee1() [ "ptrauth"(i32 0, i64 0) ]
; CHECK-DAG: call void @callee2() [ "ptrauth"(i32 0, i64 0) ]

;; The still-indirect call in each mixed clone retains its ptrauth bundle.
; CHECK-DAG: call void %loaded() [ "ptrauth"(i32 0, i64 0) ]
; CHECK-DAG: call void %loaded() [ "ptrauth"(i32 0, i64 0) ]
