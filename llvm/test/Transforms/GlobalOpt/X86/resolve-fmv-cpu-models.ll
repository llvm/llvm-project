; REQUIRES: x86-registered-target
; RUN: opt -passes=globalopt -S %s | FileCheck %s
;
; CPU predicates describe runtime identity, independently of ISA features.
; fmv-priority records the resolver's order, including source-order ties.

target triple = "x86_64-unknown-linux-gnu"
@haswell = external global i1
@v3 = external global i1
@atom = external global i1
@goldmont = external global i1

; Missing or malformed ordering must disable resolution. CPU type and subtype
; ABI values also occupy distinct namespaces (goldmont-plus and Haswell are 13).
@malformed = weak_odr ifunc i32 (i32), ptr @malformed.resolver
declare i32 @malformed_version(i32) #8
define ptr @malformed.resolver() {
  %c = load i1, ptr @haswell
  %r = select i1 %c, ptr @malformed_version, ptr @default
  ret ptr %r
}

@overflow = weak_odr ifunc i32 (i32), ptr @overflow.resolver
declare i32 @overflow_version(i32) #9
define ptr @overflow.resolver() {
  %c = load i1, ptr @haswell
  %r = select i1 %c, ptr @overflow_version, ptr @default
  ret ptr %r
}

@type13 = weak_odr ifunc i32 (i32), ptr @type13.resolver
declare i32 @type13_version(i32) #10
define ptr @type13.resolver() {
  %c = load i1, ptr @goldmont
  %r = select i1 %c, ptr @type13_version, ptr @default
  ret ptr %r
}

; Mixed old/new bitcode cannot establish the complete resolver order.
@mixed_encoding = weak_odr ifunc i32 (i32), ptr @mixed_encoding.resolver
declare i32 @legacy_v3(i32) #11
define ptr @mixed_encoding.resolver() {
  %h = load i1, ptr @haswell
  %v = load i1, ptr @v3
  %lo = select i1 %v, ptr @legacy_v3, ptr @default
  %r = select i1 %h, ptr @haswell_version, ptr %lo
  ret ptr %r
}

@unordered = weak_odr ifunc i32 (i32), ptr @unordered.resolver
@simple = weak_odr ifunc i32 (i32), ptr @simple.resolver
@simple_caller = weak_odr ifunc i32 (i32), ptr @simple_caller.resolver
@mixed = weak_odr ifunc i32 (i32), ptr @mixed.resolver
@mixed_caller = weak_odr ifunc i32 (i32), ptr @mixed_caller.resolver
@alias_callee = weak_odr ifunc i32 (i32), ptr @alias_callee.resolver
@alias_caller = weak_odr ifunc i32 (i32), ptr @alias_caller.resolver

declare i32 @default(i32) #0
declare i32 @haswell_version(i32) #1
declare i32 @v3_version(i32) #2
declare i32 @bonnell_version(i32) #3
declare i32 @unordered_haswell(i32) #7

; Without exact ordering metadata, leave a CPU-model family unresolved.
define ptr @unordered.resolver() {
  %h = load i1, ptr @haswell
  %r = select i1 %h, ptr @unordered_haswell, ptr @default
  ret ptr %r
}

define ptr @simple.resolver() {
  %h = load i1, ptr @haswell
  %r = select i1 %h, ptr @haswell_version, ptr @default
  ret ptr %r
}

define ptr @simple_caller.resolver() {
  %h = load i1, ptr @haswell
  %r = select i1 %h, ptr @simple_haswell, ptr @simple_default
  ret ptr %r
}

; Both matching clones resolve, including the default version.
define i32 @simple_haswell(i32 %x) #1 {
; CHECK-LABEL: define i32 @simple_haswell(
; CHECK: call i32 @haswell_version(
; CHECK: call i32 @unordered(
; CHECK: call i32 @malformed(
; CHECK: call i32 @overflow(
; CHECK: call i32 @type13(
; CHECK: call i32 @mixed_encoding(
  %r = call i32 @simple(i32 %x)
  %u = call i32 @unordered(i32 %r)
  %m = call i32 @malformed(i32 %u)
  %o = call i32 @overflow(i32 %m)
  %t = call i32 @type13(i32 %o)
  %e = call i32 @mixed_encoding(i32 %t)
  ret i32 %e
}

define i32 @simple_default(i32 %x) #0 {
; CHECK-LABEL: define i32 @simple_default(
; CHECK: call i32 @default(
  %r = call i32 @simple(i32 %x)
  ret i32 %r
}

; Haswell and x86-64-v3 have equal base priority. The callee lists v3 first;
; the caller lists Haswell first. Both predicates can hold at runtime.
define ptr @mixed.resolver() {
  %h = load i1, ptr @haswell
  %v = load i1, ptr @v3
  %lo = select i1 %h, ptr @haswell_version, ptr @default
  %r = select i1 %v, ptr @v3_version, ptr %lo
  ret ptr %r
}

define ptr @mixed_caller.resolver() {
  %h = load i1, ptr @haswell
  %v = load i1, ptr @v3
  %lo = select i1 %v, ptr @mixed_v3, ptr @mixed_default
  %r = select i1 %h, ptr @mixed_haswell, ptr %lo
  ret ptr %r
}

define i32 @mixed_haswell(i32 %x) #4 {
; CHECK-LABEL: define i32 @mixed_haswell(
; CHECK: call i32 @mixed(
  %r = call i32 @mixed(i32 %x)
  ret i32 %r
}

define i32 @mixed_v3(i32 %x) #5 {
; CHECK-LABEL: define i32 @mixed_v3(
; CHECK: call i32 @v3_version(
  %r = call i32 @mixed(i32 %x)
  ret i32 %r
}

define i32 @mixed_default(i32 %x) #0 {
; CHECK-LABEL: define i32 @mixed_default(
; CHECK: call i32 @default(
  %r = call i32 @mixed(i32 %x)
  ret i32 %r
}

; Alias spellings refer to the same runtime CPU field and ABI value.
define ptr @alias_callee.resolver() {
  %a = load i1, ptr @atom
  %r = select i1 %a, ptr @bonnell_version, ptr @default
  ret ptr %r
}

define ptr @alias_caller.resolver() {
  %a = load i1, ptr @atom
  %r = select i1 %a, ptr @atom_caller, ptr @alias_default
  ret ptr %r
}

define i32 @atom_caller(i32 %x) #6 {
; CHECK-LABEL: define i32 @atom_caller(
; CHECK: call i32 @bonnell_version(
  %r = call i32 @alias_callee(i32 %x)
  ret i32 %r
}

define i32 @alias_default(i32 %x) #0 {
; CHECK-LABEL: define i32 @alias_default(
; CHECK: call i32 @default(
  %r = call i32 @alias_callee(i32 %x)
  ret i32 %r
}

attributes #0 = { "fmv-features" "fmv-priority"="1" }
attributes #1 = { "fmv-features"="arch=haswell" "fmv-priority"="20" }
attributes #2 = { "fmv-features"="x86-64-v3" "fmv-priority"="30" }
attributes #3 = { "fmv-features"="arch=bonnell" "fmv-priority"="20" }
attributes #4 = { "fmv-features"="arch=haswell" "fmv-priority"="30" }
attributes #5 = { "fmv-features"="x86-64-v3" "fmv-priority"="20" }
attributes #6 = { "fmv-features"="arch=atom" "fmv-priority"="20" }
attributes #7 = { "fmv-features"="arch=haswell" }
attributes #8 = { "fmv-features"="arch=haswell" "fmv-priority"="invalid" }
attributes #9 = { "fmv-features"="arch=haswell" "fmv-priority"="18446744073709551616" }
attributes #10 = { "fmv-features"="arch=goldmont-plus" "fmv-priority"="20" }
attributes #11 = { "fmv-features"="x86-64-v3" }
