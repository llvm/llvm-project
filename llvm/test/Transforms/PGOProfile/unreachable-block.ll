; RUN: llvm-profdata merge %S/Inputs/unreachable-block.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S

declare ptr @bar()

define ptr @foo() {
entry:
  ret ptr null

2:
  ret ptr null
}
