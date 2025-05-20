; RUN: llvm-profdata merge %S/Inputs/unreachable-block.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s

declare ptr @bar()

; CHECK: define ptr @foo
define ptr @foo() {
entry:
  ret ptr null

2:
  ret ptr null
}
