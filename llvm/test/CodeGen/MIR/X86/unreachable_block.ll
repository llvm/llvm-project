; RUN: llc -mtriple x86_64-- -stop-before peephole-opt -o %t.mir %s
; RUN: llc -mtriple x86_64-- -run-pass none %t.mir -verify-machineinstrs -o - | FileCheck %s

; Unreachable blocks in the machine instr representation are these
; weird empty blocks with no successors.
; The MIR printer used to not print empty lists of successors. However,
; the MIR parser now treats non-printed list of successors as "please
; guess it for me". As a result, the parser tries to guess the list of
; successors and given the block is empty, just assumes it falls through
; the next block.
;
; The following test case used to fail the verifier because the false
; path ended up falling through split.true and now, the definition of
; %v does not dominate all its uses.
; Indeed, we go from the following CFG:
;          entry
;         /      \
;    true (def)   false
;        |
;  split.true (use)
;
; To this one:
;          entry
;         /      \
;    true (def)   false
;        |        /  <-- invalid edge
;  split.true (use)
;
; Because of the invalid edge, we get the "def does not
; dominate all uses" error.
;
; CHECK-LABEL: name: foo
; CHECK-LABEL: bb.{{[0-9]+}}.false:
; CHECK-NEXT: successors:
; CHECK-NOT: %bb.{{[0-9]+}}.split.true
; CHECK-LABEL: bb.{{[0-9]+}}.split.true:
define void @foo(ptr %bar) {
  br i1 undef, label %true, label %false
true:
  %v = load i32, ptr %bar
  br label %split.true
false:
  unreachable
split.true:
  %vInc = add i32 %v, 1
  store i32 %vInc, ptr %bar
  ret void
}
