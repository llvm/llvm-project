; REQUIRES: x86-registered-target
; RUN: opt -passes=globalopt -S %s | FileCheck %s
; RUN: opt -passes=globalopt -inline-threshold=0 -inline-call-penalty=0 -S %s | FileCheck %s --check-prefix=EXPENSIVE
;
; All possible resolver choices must have equivalent, small integer bodies.
; A target-compatible body can then replace an unresolved IFUNC call.

target triple = "x86_64-unknown-linux-gnu"
@features = external global i32
@effect = external global i32

declare i32 @external(i32)

@equal = weak_odr ifunc i32 (i32, i32), ptr @equal.resolver

define ptr @equal.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @equal.high, ptr @equal.low
  ret ptr %v
}

define i32 @equal.low(i32 %x, i32 %y) #0 {
  %v = xor i32 %x, 17
  ret i32 %v
}
define i32 @equal.high(i32 %x, i32 %y) #1 {
  %v = xor i32 %x, 17
  ret i32 %v
}

; CHECK-LABEL: define i32 @equal.caller(
; CHECK: call i32 @equal.low(
define i32 @equal.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @equal(i32 %x, i32 %y)
  ret i32 %v
}

@constant = weak_odr ifunc i32 (i32, i32), ptr @constant.resolver

define ptr @constant.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @constant.high, ptr @constant.low
  ret ptr %v
}

define i32 @constant.low(i32 %x, i32 %y) #0 {
  %v = xor i32 %x, 17
  ret i32 %v
}
define i32 @constant.high(i32 %x, i32 %y) #1 {
  %v = xor i32 %x, 18
  ret i32 %v
}

; CHECK-LABEL: define i32 @constant.caller(
; CHECK: call i32 @constant(
define i32 @constant.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @constant(i32 %x, i32 %y)
  ret i32 %v
}

@flags = weak_odr ifunc i32 (i32, i32), ptr @flags.resolver

define ptr @flags.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @flags.high, ptr @flags.low
  ret ptr %v
}

define i32 @flags.low(i32 %x, i32 %y) #0 {
  %v = add nsw i32 %x, 17
  ret i32 %v
}
define i32 @flags.high(i32 %x, i32 %y) #1 {
  %v = add i32 %x, 17
  ret i32 %v
}

; CHECK-LABEL: define i32 @flags.caller(
; CHECK: call i32 @flags(
define i32 @flags.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @flags(i32 %x, i32 %y)
  ret i32 %v
}

@operand = weak_odr ifunc i32 (i32, i32), ptr @operand.resolver

define ptr @operand.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @operand.high, ptr @operand.low
  ret ptr %v
}

define i32 @operand.low(i32 %x, i32 %y) #0 {
  %v = sub i32 %x, %y
  ret i32 %v
}
define i32 @operand.high(i32 %x, i32 %y) #1 {
  %v = sub i32 %y, %x
  ret i32 %v
}

; CHECK-LABEL: define i32 @operand.caller(
; CHECK: call i32 @operand(
define i32 @operand.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @operand(i32 %x, i32 %y)
  ret i32 %v
}

@effects = weak_odr ifunc i32 (i32, i32), ptr @effects.resolver

define ptr @effects.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @effects.high, ptr @effects.low
  ret ptr %v
}

define i32 @effects.low(i32 %x, i32 %y) #0 {
  store i32 %x, ptr @effect
  %v = xor i32 %x, 17
  ret i32 %v
}
define i32 @effects.high(i32 %x, i32 %y) #1 {
  store i32 %x, ptr @effect
  %v = xor i32 %x, 17
  ret i32 %v
}

; CHECK-LABEL: define i32 @effects.caller(
; CHECK: call i32 @effects(
define i32 @effects.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @effects(i32 %x, i32 %y)
  ret i32 %v
}

@calls = weak_odr ifunc i32 (i32, i32), ptr @calls.resolver

define ptr @calls.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @calls.high, ptr @calls.low
  ret ptr %v
}

define i32 @calls.low(i32 %x, i32 %y) #0 {
  %v = call i32 @external(i32 %x)
  ret i32 %v
}
define i32 @calls.high(i32 %x, i32 %y) #1 {
  %v = call i32 @external(i32 %x)
  ret i32 %v
}

; CHECK-LABEL: define i32 @calls.caller(
; CHECK: call i32 @calls(
define i32 @calls.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @calls(i32 %x, i32 %y)
  ret i32 %v
}

@preemptible = weak_odr ifunc i32 (i32, i32), ptr @preemptible.resolver

define ptr @preemptible.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @preemptible.high, ptr @preemptible.low
  ret ptr %v
}

define weak i32 @preemptible.low(i32 %x, i32 %y) #0 {
  %v = xor i32 %x, 17
  ret i32 %v
}
define i32 @preemptible.high(i32 %x, i32 %y) #1 {
  %v = xor i32 %x, 17
  ret i32 %v
}

; CHECK-LABEL: define i32 @preemptible.caller(
; CHECK: call i32 @preemptible(
define i32 @preemptible.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @preemptible(i32 %x, i32 %y)
  ret i32 %v
}

@attrs = weak_odr ifunc i32 (i32, i32), ptr @attrs.resolver

define ptr @attrs.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @attrs.high, ptr @attrs.low
  ret ptr %v
}

define i32 @attrs.low(i32 %x, i32 %y) #0 {
  %v = xor i32 %x, 17
  ret i32 %v
}
define zeroext i32 @attrs.high(i32 %x, i32 %y) #1 {
  %v = xor i32 %x, 17
  ret i32 %v
}

; CHECK-LABEL: define i32 @attrs.caller(
; CHECK: call i32 @attrs(
define i32 @attrs.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @attrs(i32 %x, i32 %y)
  ret i32 %v
}

@poison = weak_odr ifunc i32 (i32, i32), ptr @poison.resolver

define ptr @poison.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @poison.high, ptr @poison.low
  ret ptr %v
}

define i32 @poison.low(i32 %x, i32 %y) #0 {
  ret i32 poison
}
define i32 @poison.high(i32 %x, i32 %y) #1 {
  ret i32 poison
}

; CHECK-LABEL: define i32 @poison.caller(
; CHECK: call i32 @poison(
define i32 @poison.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @poison(i32 %x, i32 %y)
  ret i32 %v
}

@noinline = weak_odr ifunc i32 (i32, i32), ptr @noinline.resolver

define ptr @noinline.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @noinline.high, ptr @noinline.low
  ret ptr %v
}

define i32 @noinline.low(i32 %x, i32 %y) #3 {
  %v = xor i32 %x, 17
  ret i32 %v
}
define i32 @noinline.high(i32 %x, i32 %y) #1 {
  %v = xor i32 %x, 17
  ret i32 %v
}

; CHECK-LABEL: define i32 @noinline.caller(
; CHECK: call i32 @noinline(
define i32 @noinline.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @noinline(i32 %x, i32 %y)
  ret i32 %v
}

; A declaration cannot establish equivalence with all runtime choices.
@unknown = weak_odr ifunc i32 (i32, i32), ptr @unknown.resolver
define ptr @unknown.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @unknown.high, ptr @equal.low
  ret ptr %v
}
declare i32 @unknown.high(i32, i32) #1
; CHECK-LABEL: define i32 @unknown.caller(
; CHECK: call i32 @unknown(
define i32 @unknown.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @unknown(i32 %x, i32 %y)
  ret i32 %v
}

; Do not bypass callsite restrictions or rewrite an interposable IFUNC.
@preemptible_ifunc = weak ifunc i32 (i32, i32), ptr @equal.resolver
; CHECK-LABEL: define i32 @restricted(
; CHECK: call i32 @equal(
; CHECK: call i32 @preemptible_ifunc(
define i32 @restricted(i32 %x, i32 %y) #2 {
  %a = call i32 @equal(i32 %x, i32 %y) #4
  %b = call i32 @preemptible_ifunc(i32 %a, i32 %y)
  ret i32 %b
}

attributes #0 = { "fmv-features" "target-features"="+sse2" }
attributes #1 = { "fmv-features"="avx512f" "target-features"="+avx512f" }
attributes #2 = { "target-features"="+avx2" }
attributes #3 = { noinline "fmv-features" "target-features"="+sse2" }
attributes #4 = { noinline }

; Exact resolution can select a body incompatible with the caller's codegen
; settings. A runtime predicate and codegen features need not be identical.
@known = weak_odr ifunc i32 (i32, i32), ptr @known.resolver
@normal = weak_odr ifunc i32 (i32, i32), ptr @normal.resolver

define ptr @known.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @known.high, ptr @known.low
  ret ptr %v
}
define i32 @known.low(i32 %x, i32 %y) #0 {
  ret i32 %x
}
; CHECK-LABEL: define i32 @known.high(
; CHECK: call i32 @equal.low(
define i32 @known.high(i32 %x, i32 %y) #5 {
  %v = call i32 @equal(i32 %x, i32 %y)
  ret i32 %v
}
define ptr @normal.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @normal.high, ptr @known.low
  ret ptr %v
}
; Keep the exact resolver choice when it is already inline-compatible.
; CHECK-LABEL: define i32 @normal.high(
; CHECK: call i32 @equal.high(
define i32 @normal.high(i32 %x, i32 %y) #1 {
  %v = call i32 @equal(i32 %x, i32 %y)
  ret i32 %v
}
attributes #5 = { "fmv-features"="avx512f" "target-features"="+avx2" }

; A compatible equivalent body must also pass the inline cost model.
@costly = weak_odr ifunc i32 (i32), ptr @costly.resolver
define ptr @costly.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @costly.high, ptr @costly.low
  ret ptr %v
}
define i32 @costly.low(i32 %x) #0 {
  %v0 = mul i32 %x, 101
  %v1 = mul i32 %v0, 103
  %v2 = mul i32 %v1, 105
  %v3 = mul i32 %v2, 107
  %v4 = mul i32 %v3, 109
  %v5 = mul i32 %v4, 111
  %v6 = mul i32 %v5, 113
  %v7 = mul i32 %v6, 115
  %v8 = mul i32 %v7, 117
  %v9 = mul i32 %v8, 119
  %v10 = mul i32 %v9, 121
  %v11 = mul i32 %v10, 123
  %v12 = mul i32 %v11, 125
  %v13 = mul i32 %v12, 127
  %v14 = mul i32 %v13, 129
  %v15 = mul i32 %v14, 131
  %v16 = mul i32 %v15, 133
  %v17 = mul i32 %v16, 135
  %v18 = mul i32 %v17, 137
  %v19 = mul i32 %v18, 139
  ret i32 %v19
}
define i32 @costly.high(i32 %x) #1 {
  %v0 = mul i32 %x, 101
  %v1 = mul i32 %v0, 103
  %v2 = mul i32 %v1, 105
  %v3 = mul i32 %v2, 107
  %v4 = mul i32 %v3, 109
  %v5 = mul i32 %v4, 111
  %v6 = mul i32 %v5, 113
  %v7 = mul i32 %v6, 115
  %v8 = mul i32 %v7, 117
  %v9 = mul i32 %v8, 119
  %v10 = mul i32 %v9, 121
  %v11 = mul i32 %v10, 123
  %v12 = mul i32 %v11, 125
  %v13 = mul i32 %v12, 127
  %v14 = mul i32 %v13, 129
  %v15 = mul i32 %v14, 131
  %v16 = mul i32 %v15, 133
  %v17 = mul i32 %v16, 135
  %v18 = mul i32 %v17, 137
  %v19 = mul i32 %v18, 139
  ret i32 %v19
}
; CHECK-LABEL: define i32 @costly.caller(
; CHECK: call i32 @costly.low(
; EXPENSIVE-LABEL: define i32 @costly.caller(
; EXPENSIVE: call i32 @costly(
define i32 @costly.caller(i32 %x) #2 {
  %v = call i32 @costly(i32 %x)
  ret i32 %v
}

; Floating-point bodies need stronger target-semantic reasoning.
@floating = weak_odr ifunc float (float), ptr @floating.resolver
define ptr @floating.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @floating.high, ptr @floating.low
  ret ptr %v
}
define float @floating.low(float %x) #0 {
  %v = fadd float %x, 1.0
  ret float %v
}
define float @floating.high(float %x) #1 {
  %v = fadd float %x, 1.0
  ret float %v
}
; CHECK-LABEL: define float @floating.caller(
; CHECK: call float @floating(
define float @floating.caller(float %x) #2 {
  %v = call float @floating(float %x)
  ret float %v
}

; Preserve unsupported callsite contracts even for equivalent versions.
; CHECK-LABEL: define i32 @bundled.caller(
; CHECK: call i32 @equal(i32 %x, i32 %y) [ "deopt"(i32 7) ]
define i32 @bundled.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @equal(i32 %x, i32 %y) [ "deopt"(i32 7) ]
  ret i32 %v
}
; CHECK-LABEL: define i32 @musttail.caller(
; CHECK: musttail call i32 @equal(
define i32 @musttail.caller(i32 %x, i32 %y) #2 {
  %v = musttail call i32 @equal(i32 %x, i32 %y)
  ret i32 %v
}
; CHECK-LABEL: define i32 @optnone.caller(
; CHECK: call i32 @equal(
define i32 @optnone.caller(i32 %x, i32 %y) #6 {
  %v = call i32 @equal(i32 %x, i32 %y)
  ret i32 %v
}
attributes #6 = { noinline optnone "target-features"="+avx2" }

; Even equal instruction metadata needs an explicit equivalence policy.
@annotated = weak_odr ifunc i32 (i32, i32), ptr @annotated.resolver
define ptr @annotated.resolver() {
  %f = load i32, ptr @features
  %c = icmp ne i32 %f, 0
  %v = select i1 %c, ptr @annotated.high, ptr @annotated.low
  ret ptr %v
}
define i32 @annotated.low(i32 %x, i32 %y) #0 {
  %v = xor i32 %x, 17, !annotation !0
  ret i32 %v
}
define i32 @annotated.high(i32 %x, i32 %y) #1 {
  %v = xor i32 %x, 17, !annotation !0
  ret i32 %v
}
; CHECK-LABEL: define i32 @annotated.caller(
; CHECK: call i32 @annotated(
define i32 @annotated.caller(i32 %x, i32 %y) #2 {
  %v = call i32 @annotated(i32 %x, i32 %y)
  ret i32 %v
}
!0 = !{!"retain"}
