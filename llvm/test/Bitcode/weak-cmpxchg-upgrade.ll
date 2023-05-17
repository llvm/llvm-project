; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

; cmpxchg-upgrade.ll.bc was produced by running a version of llvm-as from just
; before the IR change on this file.

define i32 @test(i32* %addr, i32 %old, i32 %new) {
; CHECK:  [[TMP:%.*]] = cmpxchg ptr %addr, i32 %old, i32 %new seq_cst monotonic
; CHECK:  %val = extractvalue { i32, i1 } [[TMP]], 0
  %val = cmpxchg i32* %addr, i32 %old, i32 %new seq_cst monotonic
  ret i32 %val
}

define i32 @test(i32* %addr, i32 %old, i32 %new) {
  ret i1 %val
}
