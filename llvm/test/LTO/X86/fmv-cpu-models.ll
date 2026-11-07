; REQUIRES: x86-registered-target
; RUN: split-file %s %t
; RUN: llvm-as %t/caller.ll -o %t/caller.bc
; RUN: llvm-as %t/callee.ll -o %t/callee.bc
; RUN: llvm-lto2 run %t/caller.bc %t/callee.bc -o %t/full -save-temps \
; RUN:   -r=%t/caller.bc,haswell,l \
; RUN:   -r=%t/caller.bc,v3,l \
; RUN:   -r=%t/caller.bc,caller_matching,plx \
; RUN:   -r=%t/caller.bc,caller_tied,plx \
; RUN:   -r=%t/caller.bc,caller_matching.default,plx \
; RUN:   -r=%t/caller.bc,caller_matching.haswell,plx \
; RUN:   -r=%t/caller.bc,caller_matching.resolver,plx \
; RUN:   -r=%t/caller.bc,caller_tied.default,plx \
; RUN:   -r=%t/caller.bc,caller_tied.haswell,plx \
; RUN:   -r=%t/caller.bc,caller_tied.v3,plx \
; RUN:   -r=%t/caller.bc,caller_tied.resolver,plx \
; RUN:   -r=%t/caller.bc,callee_matching,l \
; RUN:   -r=%t/caller.bc,callee_tied,l \
; RUN:   -r=%t/callee.bc,haswell,l \
; RUN:   -r=%t/callee.bc,v3,l \
; RUN:   -r=%t/callee.bc,callee_matching,plx \
; RUN:   -r=%t/callee.bc,callee_tied,plx \
; RUN:   -r=%t/callee.bc,callee_matching.default,plx \
; RUN:   -r=%t/callee.bc,callee_matching.haswell,plx \
; RUN:   -r=%t/callee.bc,callee_matching.resolver,plx \
; RUN:   -r=%t/callee.bc,callee_tied.default,plx \
; RUN:   -r=%t/callee.bc,callee_tied.haswell,plx \
; RUN:   -r=%t/callee.bc,callee_tied.v3,plx \
; RUN:   -r=%t/callee.bc,callee_tied.resolver,plx
; RUN: llvm-dis %t/full.0.4.opt.bc -o - | FileCheck %s --check-prefix=FULL
; RUN: opt -module-summary %t/caller.ll -o %t/caller.bc
; RUN: opt -module-summary %t/callee.ll -o %t/callee.bc
; RUN: llvm-lto2 run %t/caller.bc %t/callee.bc -o %t/thin -save-temps \
; RUN:   -r=%t/caller.bc,haswell,l \
; RUN:   -r=%t/caller.bc,v3,l \
; RUN:   -r=%t/caller.bc,caller_matching,plx \
; RUN:   -r=%t/caller.bc,caller_tied,plx \
; RUN:   -r=%t/caller.bc,caller_matching.default,plx \
; RUN:   -r=%t/caller.bc,caller_matching.haswell,plx \
; RUN:   -r=%t/caller.bc,caller_matching.resolver,plx \
; RUN:   -r=%t/caller.bc,caller_tied.default,plx \
; RUN:   -r=%t/caller.bc,caller_tied.haswell,plx \
; RUN:   -r=%t/caller.bc,caller_tied.v3,plx \
; RUN:   -r=%t/caller.bc,caller_tied.resolver,plx \
; RUN:   -r=%t/caller.bc,callee_matching,l \
; RUN:   -r=%t/caller.bc,callee_tied,l \
; RUN:   -r=%t/callee.bc,haswell,l \
; RUN:   -r=%t/callee.bc,v3,l \
; RUN:   -r=%t/callee.bc,callee_matching,plx \
; RUN:   -r=%t/callee.bc,callee_tied,plx \
; RUN:   -r=%t/callee.bc,callee_matching.default,plx \
; RUN:   -r=%t/callee.bc,callee_matching.haswell,plx \
; RUN:   -r=%t/callee.bc,callee_matching.resolver,plx \
; RUN:   -r=%t/callee.bc,callee_tied.default,plx \
; RUN:   -r=%t/callee.bc,callee_tied.haswell,plx \
; RUN:   -r=%t/callee.bc,callee_tied.v3,plx \
; RUN:   -r=%t/callee.bc,callee_tied.resolver,plx
; RUN: llvm-dis %t/thin.1.4.opt.bc -o - | FileCheck %s --check-prefix=UNRESOLVED
; RUN: llvm-as %t/caller.ll -o %t/caller.bc
; RUN: sed 's/"fmv-priority"="1"//' %t/callee.ll | llvm-as -o %t/callee.bc
; RUN: llvm-lto2 run %t/caller.bc %t/callee.bc -o %t/mixed -save-temps \
; RUN:   -r=%t/caller.bc,haswell,l \
; RUN:   -r=%t/caller.bc,v3,l \
; RUN:   -r=%t/caller.bc,caller_matching,plx \
; RUN:   -r=%t/caller.bc,caller_tied,plx \
; RUN:   -r=%t/caller.bc,caller_matching.default,plx \
; RUN:   -r=%t/caller.bc,caller_matching.haswell,plx \
; RUN:   -r=%t/caller.bc,caller_matching.resolver,plx \
; RUN:   -r=%t/caller.bc,caller_tied.default,plx \
; RUN:   -r=%t/caller.bc,caller_tied.haswell,plx \
; RUN:   -r=%t/caller.bc,caller_tied.v3,plx \
; RUN:   -r=%t/caller.bc,caller_tied.resolver,plx \
; RUN:   -r=%t/caller.bc,callee_matching,l \
; RUN:   -r=%t/caller.bc,callee_tied,l \
; RUN:   -r=%t/callee.bc,haswell,l \
; RUN:   -r=%t/callee.bc,v3,l \
; RUN:   -r=%t/callee.bc,callee_matching,plx \
; RUN:   -r=%t/callee.bc,callee_tied,plx \
; RUN:   -r=%t/callee.bc,callee_matching.default,plx \
; RUN:   -r=%t/callee.bc,callee_matching.haswell,plx \
; RUN:   -r=%t/callee.bc,callee_matching.resolver,plx \
; RUN:   -r=%t/callee.bc,callee_tied.default,plx \
; RUN:   -r=%t/callee.bc,callee_tied.haswell,plx \
; RUN:   -r=%t/callee.bc,callee_tied.v3,plx \
; RUN:   -r=%t/callee.bc,callee_tied.resolver,plx
; RUN: llvm-dis %t/mixed.0.4.opt.bc -o - | FileCheck %s --check-prefix=UNRESOLVED
;
; Full LTO resolves and inlines matching CPU-model clones across modules.
; Reversed Haswell/v3 tie order must retain dispatch where either callee can win.
; ThinLTO retains the cross-module IFUNC calls; metadata must survive without
; changing their semantics or producing invalid IR. Mixed legacy/new priority
; encodings must also retain dispatch after full LTO linking.
;
; FULL-LABEL: define{{.*}} i32 @caller_matching.default(
; FULL-NOT: call
; FULL: add i32 %x, 1
; FULL-NEXT: ret i32
; UNRESOLVED-LABEL: define{{.*}} i32 @caller_matching.default(
; UNRESOLVED: call i32 @callee_matching(
; FULL-LABEL: define{{.*}} i32 @caller_matching.haswell(
; FULL-NOT: call
; FULL: add i32 %x, 2
; FULL-NEXT: ret i32
; UNRESOLVED-LABEL: define{{.*}} i32 @caller_matching.haswell(
; UNRESOLVED: call i32 @callee_matching(
; FULL-LABEL: define{{.*}} i32 @caller_tied.default(
; FULL-NOT: call
; FULL: add i32 %x, 1
; FULL-NEXT: ret i32
; UNRESOLVED-LABEL: define{{.*}} i32 @caller_tied.default(
; UNRESOLVED: call i32 @callee_tied(
; FULL-LABEL: define{{.*}} i32 @caller_tied.haswell(
; FULL: call i32 @callee_tied(
; UNRESOLVED-LABEL: define{{.*}} i32 @caller_tied.haswell(
; UNRESOLVED: call i32 @callee_tied(
; FULL-LABEL: define{{.*}} i32 @caller_tied.v3(
; FULL-NOT: call
; FULL: add i32 %x, 3
; FULL-NEXT: ret i32
; UNRESOLVED-LABEL: define{{.*}} i32 @caller_tied.v3(
; UNRESOLVED: call i32 @callee_tied(

;--- caller.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@haswell = external global i1
@v3 = external global i1
@caller_matching = weak_odr ifunc i32 (i32), ptr @caller_matching.resolver
declare i32 @callee_matching(i32)

define i32 @caller_matching.default(i32 %x) #0 {
  %r = call i32 @callee_matching(i32 %x)
  ret i32 %r
}

define i32 @caller_matching.haswell(i32 %x) #1 {
  %r = call i32 @callee_matching(i32 %x)
  ret i32 %r
}

define ptr @caller_matching.resolver() {
  %h = load i1, ptr @haswell
  %r = select i1 %h, ptr @caller_matching.haswell, ptr @caller_matching.default
  ret ptr %r
}
@caller_tied = weak_odr ifunc i32 (i32), ptr @caller_tied.resolver
declare i32 @callee_tied(i32)

define i32 @caller_tied.default(i32 %x) #0 {
  %r = call i32 @callee_tied(i32 %x)
  ret i32 %r
}

define i32 @caller_tied.haswell(i32 %x) #1 {
  %r = call i32 @callee_tied(i32 %x)
  ret i32 %r
}

define i32 @caller_tied.v3(i32 %x) #2 {
  %r = call i32 @callee_tied(i32 %x)
  ret i32 %r
}

define ptr @caller_tied.resolver() {
  %h = load i1, ptr @haswell
  %v = load i1, ptr @v3
  %lo = select i1 %v, ptr @caller_tied.v3, ptr @caller_tied.default
  %r = select i1 %h, ptr @caller_tied.haswell, ptr %lo
  ret ptr %r
}

attributes #0 = { "fmv-features" "fmv-priority"="1" }
attributes #1 = { "fmv-features"="arch=haswell" "fmv-priority"="30" }
attributes #2 = { "fmv-features"="x86-64-v3" "fmv-priority"="20" }
attributes #3 = { "fmv-features"="arch=haswell" "fmv-priority"="20" }
attributes #4 = { "fmv-features"="x86-64-v3" "fmv-priority"="30" }

;--- callee.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@haswell = external global i1
@v3 = external global i1
@callee_matching = weak_odr ifunc i32 (i32), ptr @callee_matching.resolver

define i32 @callee_matching.default(i32 %x) #0 {
  %r = add i32 %x, 1
  ret i32 %r
}

define i32 @callee_matching.haswell(i32 %x) #1 {
  %r = add i32 %x, 2
  ret i32 %r
}

define ptr @callee_matching.resolver() {
  %h = load i1, ptr @haswell
  %r = select i1 %h, ptr @callee_matching.haswell, ptr @callee_matching.default
  ret ptr %r
}
@callee_tied = weak_odr ifunc i32 (i32), ptr @callee_tied.resolver

define i32 @callee_tied.default(i32 %x) #0 {
  %r = add i32 %x, 1
  ret i32 %r
}

define i32 @callee_tied.haswell(i32 %x) #3 {
  %r = add i32 %x, 2
  ret i32 %r
}

define i32 @callee_tied.v3(i32 %x) #4 {
  %r = add i32 %x, 3
  ret i32 %r
}

define ptr @callee_tied.resolver() {
  %h = load i1, ptr @haswell
  %v = load i1, ptr @v3
  %lo = select i1 %h, ptr @callee_tied.haswell, ptr @callee_tied.default
  %r = select i1 %v, ptr @callee_tied.v3, ptr %lo
  ret ptr %r
}

attributes #0 = { "fmv-features" "fmv-priority"="1" }
attributes #1 = { "fmv-features"="arch=haswell" "fmv-priority"="30" }
attributes #2 = { "fmv-features"="x86-64-v3" "fmv-priority"="20" }
attributes #3 = { "fmv-features"="arch=haswell" "fmv-priority"="20" }
attributes #4 = { "fmv-features"="x86-64-v3" "fmv-priority"="30" }
