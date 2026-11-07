; REQUIRES: x86-registered-target
; RUN: opt -passes='default<O2>' -S %s | FileCheck %s
;
; Matching clone families allow inlining, including calls through a local
; function pointer and x86-64 ISA-level predicates. Equivalent integer bodies
; also allow callers with fewer clones and a global initializer to inline.
; CPU-model versions without dispatch metadata still retain runtime dispatch.
; The resolvers use separate bits for AVX2, SSE4.2 and the x86-64-v3 predicate.

target triple = "x86_64-unknown-linux-gnu"

; CHECK: @dynamic_init = {{.*}}global i32 1
; CHECK: @llvm.global_ctors = appending global [0 x
@dynamic_init = global i32 0, align 4
@features = external global i32
@cpu_model = external global i32
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @global_init, ptr null }]

@callee = weak_odr ifunc i32 (i32), ptr @callee.resolver
@caller = weak_odr ifunc i32 (i32), ptr @caller.resolver
@limited = weak_odr ifunc i32 (i32), ptr @limited.resolver
@widen = weak_odr ifunc double (half), ptr @widen.resolver
@wide_caller = weak_odr ifunc double (half), ptr @wide_caller.resolver
@cpu_callee = weak_odr ifunc i32 (i32), ptr @cpu_callee.resolver
@cpu_caller = weak_odr ifunc i32 (i32), ptr @cpu_caller.resolver

define i32 @callee.default.2(i32 %x) #0 {
entry:
  %add = add nsw i32 %x, 1
  ret i32 %add
}

define i32 @callee.sse4.2.0(i32 %x) #1 {
entry:
  %add = add nsw i32 %x, 1
  ret i32 %add
}

define i32 @callee.avx2.1(i32 %x) #2 {
entry:
  %add = add nsw i32 %x, 1
  ret i32 %add
}

define ptr @callee.resolver() {
  %features = load i32, ptr @features
  %mask0 = and i32 %features, 1
  %cond0 = icmp ne i32 %mask0, 0
  %mask1 = and i32 %features, 2
  %cond1 = icmp ne i32 %mask1, 0
  %choice1 = select i1 %cond1, ptr @callee.sse4.2.0, ptr @callee.default.2
  %choice0 = select i1 %cond0, ptr @callee.avx2.1, ptr %choice1
  ret ptr %choice0
}

; CHECK-LABEL: define{{.*}} i32 @caller.avx2.0(
; CHECK-NOT: call
; CHECK: add nsw i32
; CHECK-NEXT: ret i32
define i32 @caller.avx2.0(i32 %x) #2 {
entry:
  %p = alloca ptr
  store ptr @callee, ptr %p
  %fn = load ptr, ptr %p
  %call = call i32 %fn(i32 %x)
  ret i32 %call
}

; CHECK-LABEL: define{{.*}} i32 @caller.default.2(
; CHECK-NOT: call
; CHECK: add nsw i32
; CHECK-NEXT: ret i32
define i32 @caller.default.2(i32 %x) #0 {
entry:
  %p = alloca ptr
  store ptr @callee, ptr %p
  %fn = load ptr, ptr %p
  %call = call i32 %fn(i32 %x)
  ret i32 %call
}

; CHECK-LABEL: define{{.*}} i32 @caller.sse4.2.1(
; CHECK-NOT: call
; CHECK: add nsw i32
; CHECK-NEXT: ret i32
define i32 @caller.sse4.2.1(i32 %x) #1 {
entry:
  %p = alloca ptr
  store ptr @callee, ptr %p
  %fn = load ptr, ptr %p
  %call = call i32 %fn(i32 %x)
  ret i32 %call
}

define ptr @caller.resolver() {
  %features = load i32, ptr @features
  %mask0 = and i32 %features, 1
  %cond0 = icmp ne i32 %mask0, 0
  %mask1 = and i32 %features, 2
  %cond1 = icmp ne i32 %mask1, 0
  %choice1 = select i1 %cond1, ptr @caller.sse4.2.1, ptr @caller.default.2
  %choice0 = select i1 %cond0, ptr @caller.avx2.0, ptr %choice1
  ret ptr %choice0
}

; CHECK-LABEL: define{{.*}} i32 @limited.default.1(
; CHECK-NOT: call
; CHECK: add nsw i32
; CHECK-NEXT: ret i32
define i32 @limited.default.1(i32 %x) #0 {
entry:
  %call = call i32 @callee(i32 %x)
  ret i32 %call
}

; CHECK-LABEL: define{{.*}} i32 @limited.sse4.2.0(
; CHECK-NOT: call
; CHECK: add nsw i32
; CHECK-NEXT: ret i32
define i32 @limited.sse4.2.0(i32 %x) #1 {
entry:
  %call = call i32 @callee(i32 %x)
  ret i32 %call
}

define ptr @limited.resolver() {
  %features = load i32, ptr @features
  %mask0 = and i32 %features, 2
  %cond0 = icmp ne i32 %mask0, 0
  %choice0 = select i1 %cond0, ptr @limited.sse4.2.0, ptr @limited.default.1
  ret ptr %choice0
}

define double @widen.default.1(half %x) #0 {
entry:
  %conv = fpext half %x to double
  ret double %conv
}

define double @widen.arch_x86-64-v3.0(half %x) #5 {
entry:
  %conv = fpext half %x to double
  ret double %conv
}

define ptr @widen.resolver() {
  %features = load i32, ptr @features
  %mask0 = and i32 %features, 4
  %cond0 = icmp ne i32 %mask0, 0
  %choice0 = select i1 %cond0, ptr @widen.arch_x86-64-v3.0, ptr @widen.default.1
  ret ptr %choice0
}

; CHECK-LABEL: define{{.*}} double @wide_caller.default.1(
; CHECK-NOT: call
; CHECK: fpext half
; CHECK-NEXT: ret double
define double @wide_caller.default.1(half %x) #0 {
entry:
  %call = call double @widen(half %x)
  ret double %call
}

; CHECK-LABEL: define{{.*}} double @wide_caller.arch_x86-64-v3.0(
; CHECK-NOT: call
; CHECK: fpext half
; CHECK-NEXT: ret double
define double @wide_caller.arch_x86-64-v3.0(half %x) #5 {
entry:
  %call = call double @widen(half %x)
  ret double %call
}

define ptr @wide_caller.resolver() {
  %features = load i32, ptr @features
  %mask0 = and i32 %features, 4
  %cond0 = icmp ne i32 %mask0, 0
  %choice0 = select i1 %cond0, ptr @wide_caller.arch_x86-64-v3.0, ptr @wide_caller.default.1
  ret ptr %choice0
}

define i32 @cpu_callee.default.1(i32 %x) #0 {
entry:
  %add = add nsw i32 %x, 2
  ret i32 %add
}

define i32 @cpu_callee.arch_haswell.0(i32 %x) #6 {
entry:
  %add = add nsw i32 %x, 2
  ret i32 %add
}

define ptr @cpu_callee.resolver() {
  %cpu = load i32, ptr @cpu_model
  %haswell = icmp eq i32 %cpu, 13
  %result = select i1 %haswell, ptr @cpu_callee.arch_haswell.0, ptr @cpu_callee.default.1
  ret ptr %result
}

; CHECK-LABEL: define{{.*}} i32 @cpu_caller.default.1(
; CHECK: call i32 @cpu_callee(
define i32 @cpu_caller.default.1(i32 %x) #0 {
entry:
  %call = call i32 @cpu_callee(i32 %x)
  ret i32 %call
}

; CHECK-LABEL: define{{.*}} i32 @cpu_caller.arch_haswell.0(
; CHECK: call i32 @cpu_callee(
define i32 @cpu_caller.arch_haswell.0(i32 %x) #6 {
entry:
  %call = call i32 @cpu_callee(i32 %x)
  ret i32 %call
}

define ptr @cpu_caller.resolver() {
  %cpu = load i32, ptr @cpu_model
  %haswell = icmp eq i32 %cpu, 13
  %result = select i1 %haswell, ptr @cpu_caller.arch_haswell.0, ptr @cpu_caller.default.1
  ret ptr %result
}

define internal void @__cxx_global_var_init() {
entry:
  %call = call i32 @callee(i32 0)
  store i32 %call, ptr @dynamic_init, align 4
  ret void
}

; The equivalent helper bodies let GlobalOpt fold this initializer to 1.
define internal void @global_init() {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { "fmv-features" }
attributes #1 = { "fmv-features"="sse4.2" }
attributes #2 = { "fmv-features"="avx2" }
attributes #5 = { "fmv-features"="x86-64-v3" }
attributes #6 = { "target-cpu"="haswell" }
