; REQUIRES: aarch64-registered-target

; RUN: opt --passes=globalopt -o - -S < %s | FileCheck %s

; `noipa` resolvers are not eligible for inspection

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

$callee.resolver = comdat any
$caller.resolver = comdat any

@__aarch64_cpu_features = external local_unnamed_addr global { i64 }

@callee = weak_odr ifunc i32 (), ptr @callee.resolver
@caller = weak_odr ifunc i32 (), ptr @caller.resolver

declare void @__init_cpu_features_resolver() local_unnamed_addr

declare i32 @callee.default() #0
declare i32 @callee._Msve() #1
declare i32 @callee._Msve2() #2

define weak_odr ptr @callee.resolver() noipa comdat {
resolver_entry:
  tail call void @__init_cpu_features_resolver()
  %0 = load i64, ptr @__aarch64_cpu_features, align 8
  %1 = and i64 %0, 69793284352
  %2 = icmp eq i64 %1, 69793284352
  %3 = and i64 %0, 1073807616
  %4 = icmp eq i64 %3, 1073807616
  %callee._Msve.callee.default = select i1 %4, ptr @callee._Msve, ptr @callee.default
  %common.ret.op = select i1 %2, ptr @callee._Msve2, ptr %callee._Msve.callee.default
  ret ptr %common.ret.op
}

; CHECK: define i32 @caller._Msve()
; CHECK: tail call i32 @callee()
define i32 @caller._Msve() #1 {
entry:
  %call = tail call i32 @callee()
  ret i32 %call
}

; CHECK: define i32 @caller._Msve2()
; CHECK: tail call i32 @callee()
define i32 @caller._Msve2() #2 {
entry:
  %call = tail call i32 @callee()
  ret i32 %call
}

; CHECK: define i32 @caller.default()
; CHECK: tail call i32 @callee()
define i32 @caller.default() #0 {
entry:
  %call = tail call i32 @callee()
  ret i32 %call
}

define weak_odr ptr @caller.resolver() comdat {
resolver_entry:
  tail call void @__init_cpu_features_resolver()
  %0 = load i64, ptr @__aarch64_cpu_features, align 8
  %1 = and i64 %0, 69793284352
  %2 = icmp eq i64 %1, 69793284352
  %3 = and i64 %0, 1073807616
  %4 = icmp eq i64 %3, 1073807616
  %caller._Msve.caller.default = select i1 %4, ptr @caller._Msve, ptr @caller.default
  %common.ret.op = select i1 %2, ptr @caller._Msve2, ptr %caller._Msve.caller.default
  ret ptr %common.ret.op
}

attributes #0 = { "fmv-features" }
attributes #1 = { "fmv-features"="sve" }
attributes #2 = { "fmv-features"="sve2" }
