; NOTE: This test uses separate bits for the runtime predicates, including the
; x86-64-v3 level. They are not expanded sets of code-generation features.
; REQUIRES: x86-registered-target
; RUN: opt -passes=globalopt -S %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@features = external global i32
@callee = weak_odr ifunc i32 (i32), ptr @callee.resolver
@caller = weak_odr ifunc i32 (i32), ptr @caller.resolver
@limited = weak_odr ifunc i32 (i32), ptr @limited.resolver
@preemptible = weak ifunc i32 (i32), ptr @callee.resolver
@cpu_callee = weak_odr ifunc i32 (i32), ptr @cpu_callee.resolver

define ptr @callee.resolver() {
  %features = load i32, ptr @features
  %avx2 = icmp ne i32 %features, 0
  %result = select i1 %avx2, ptr @callee.avx2, ptr @callee.default
  ret ptr %result
}

define ptr @caller.resolver() {
  %features = load i32, ptr @features
  %avx2 = icmp ne i32 %features, 0
  %result = select i1 %avx2, ptr @caller.avx2, ptr @caller.default
  ret ptr %result
}

define ptr @limited.resolver() {
  ret ptr @limited.default
}

define ptr @cpu_callee.resolver() {
  %features = load i32, ptr @features
  %cpu = icmp ne i32 %features, 0
  %result = select i1 %cpu, ptr @callee.cpu, ptr @callee.default
  ret ptr %result
}

declare i32 @callee.avx2(i32) #1
declare i32 @callee.default(i32) #0
declare i32 @callee.cpu(i32) #2

define i32 @caller.avx2(i32 %x) #1 {
; CHECK-LABEL: define i32 @caller.avx2(
; CHECK: call i32 @callee.avx2(
; CHECK: call i32 @preemptible(
; CHECK: call i32 @cpu_callee(
  %a = call i32 @callee(i32 %x)
  %b = call i32 @preemptible(i32 %a)
  %c = call i32 @cpu_callee(i32 %b)
  ret i32 %c
}

define i32 @caller.default(i32 %x) #0 {
; CHECK-LABEL: define i32 @caller.default(
; CHECK: call i32 @callee.default(
  %a = call i32 @callee(i32 %x)
  ret i32 %a
}

define i32 @limited.default(i32 %x) #0 {
; CHECK-LABEL: define i32 @limited.default(
; CHECK: call i32 @callee(
  %a = call i32 @callee(i32 %x)
  ret i32 %a
}

define i32 @ordinary(i32 %x) #3 {
; CHECK-LABEL: define i32 @ordinary(
; CHECK: call i32 @callee(
  %a = call i32 @callee(i32 %x)
  ret i32 %a
}

@levels = weak_odr ifunc i32 (i32), ptr @levels.resolver
@level_caller = weak_odr ifunc i32 (i32), ptr @level_caller.resolver

define ptr @levels.resolver() {
  %features = load i32, ptr @features
  %v3 = icmp eq i32 %features, 2
  %avx2 = icmp ne i32 %features, 0
  %low = select i1 %avx2, ptr @level.avx2, ptr @level.default
  %result = select i1 %v3, ptr @level.v3, ptr %low
  ret ptr %result
}

define ptr @level_caller.resolver() {
  %features = load i32, ptr @features
  %v3 = icmp eq i32 %features, 2
  %avx2 = icmp ne i32 %features, 0
  %low = select i1 %avx2, ptr @level_caller.avx2, ptr @level_caller.default
  %result = select i1 %v3, ptr @level_caller.v3, ptr %low
  ret ptr %result
}

declare i32 @level.default(i32) #0
declare i32 @level.avx2(i32) #1
declare i32 @level.v3(i32) #4

define i32 @level_caller.default(i32 %x) #0 {
; CHECK-LABEL: define i32 @level_caller.default(
; CHECK: call i32 @level.default(
  %a = call i32 @levels(i32 %x)
  ret i32 %a
}

define i32 @level_caller.avx2(i32 %x) #1 {
; CHECK-LABEL: define i32 @level_caller.avx2(
; CHECK: call i32 @level.avx2(
  %a = call i32 @levels(i32 %x)
  ret i32 %a
}

define i32 @level_caller.v3(i32 %x) #4 {
; CHECK-LABEL: define i32 @level_caller.v3(
; CHECK: call i32 @level.v3(
  %a = call i32 @levels(i32 %x)
  ret i32 %a
}

attributes #0 = { "fmv-features" }
attributes #1 = { "fmv-features"="avx2" }
attributes #2 = { "target-cpu"="haswell" }
attributes #3 = { "target-features"="+avx2" }
attributes #4 = { "fmv-features"="x86-64-v3" }
