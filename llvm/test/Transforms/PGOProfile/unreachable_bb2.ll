; RUN: split-file %s %t
; RUN: llvm-profdata merge %t/a.proftext -o %t/a.profdata
; RUN: opt < %t/a.ll -passes=pgo-instr-use -pgo-test-profile-file=%t/a.profdata -S | FileCheck %s

;--- a.ll

declare ptr @bar()

; CHECK: define ptr @foo
; Ensure the profile hash matches. If it doesn't we emit the "instr_prof_hash_mismatch" metadata.
; CHECK-NOT: instr_prof_hash_mismatch
define ptr @foo() {
entry:
  ret ptr null

2:
  ret ptr null
}

;--- a.proftext
# IR level Instrumentation Flag
:ir
foo
# Func Hash:
742261418966908927
# Num Counters:
1
# Counter Values:
1
