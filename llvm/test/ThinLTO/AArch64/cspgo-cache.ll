; Verify that the ThinLTO cache key includes CSIR instrumentation mode and
; profile inputs.
;
; RUN: rm -rf %t; split-file %s %t
; RUN: opt -module-summary -module-hash %t/input.ll -o %t/input.bc
;
; Changing only CS instrumentation mode must not reuse an uninstrumented
; object. The shared-cache result must match a fresh-cache control.
; RUN: llvm-lto2 run -cache-dir=%t/gen-cache %t/input.bc -o %t/gen-off \
; RUN:   -r %t/input.bc,_foo,px -r %t/input.bc,_bar,
; RUN: llvm-lto2 run -cache-dir=%t/gen-cache -lto-cspgo-gen \
; RUN:   -lto-cspgo-profile-file=%t/default_%m.profraw \
; RUN:   %t/input.bc -o %t/gen-on-shared \
; RUN:   -r %t/input.bc,_foo,px -r %t/input.bc,_bar,
; RUN: llvm-lto2 run -cache-dir=%t/gen-fresh-cache -lto-cspgo-gen \
; RUN:   -lto-cspgo-profile-file=%t/default_%m.profraw \
; RUN:   %t/input.bc -o %t/gen-on-fresh \
; RUN:   -r %t/input.bc,_foo,px -r %t/input.bc,_bar,
; RUN: llvm-nm %t/gen-on-shared.1 | FileCheck %s --check-prefix=INSTRUMENTED
; RUN: cmp %t/gen-on-shared.1 %t/gen-on-fresh.1
;
; INSTRUMENTED: __llvm_profile_runtime
;
; Changing only the contents of a CSIR profile at a stable path must likewise
; invalidate the cache. The two profiles reverse the hot loop edge.
; RUN: llvm-profdata merge %t/hot-loop.proftext -o %t/active.profdata
; RUN: llvm-lto2 run -cache-dir=%t/use-cache -pgo-instrument-entry=false \
; RUN:   -lto-cspgo-profile-file=%t/active.profdata \
; RUN:   %t/input.bc -o %t/use-hot \
; RUN:   -r %t/input.bc,_foo,px -r %t/input.bc,_bar,
; RUN: llvm-profdata merge %t/cold-loop.proftext -o %t/active.profdata
; RUN: llvm-lto2 run -cache-dir=%t/use-cache -pgo-instrument-entry=false \
; RUN:   -lto-cspgo-profile-file=%t/active.profdata \
; RUN:   %t/input.bc -o %t/use-cold-shared \
; RUN:   -r %t/input.bc,_foo,px -r %t/input.bc,_bar,
; RUN: llvm-lto2 run -cache-dir=%t/use-fresh-cache \
; RUN:   -pgo-instrument-entry=false \
; RUN:   -lto-cspgo-profile-file=%t/active.profdata \
; RUN:   %t/input.bc -o %t/use-cold-fresh \
; RUN:   -r %t/input.bc,_foo,px -r %t/input.bc,_bar,
; RUN: cmp %t/use-cold-shared.1 %t/use-cold-fresh.1
; RUN: not cmp %t/use-hot.1 %t/use-cold-shared.1
;
;--- input.ll
source_filename = "cspgo.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios12.0.0"

define void @foo() {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %next, %for.body ]
  call void @bar(i32 %i)
  %odd = or i32 %i, 1
  call void @bar(i32 %odd)
  %next = add nuw nsw i32 %i, 2
  %cmp = icmp ult i32 %next, 200000
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

declare void @bar(i32)

;--- hot-loop.proftext
:csir
foo
# Func Hash:
1936928561113927580
# Num Counters:
2
# Counter Values:
100000
1

;--- cold-loop.proftext
:csir
foo
# Func Hash:
1936928561113927580
# Num Counters:
2
# Counter Values:
1
100000
