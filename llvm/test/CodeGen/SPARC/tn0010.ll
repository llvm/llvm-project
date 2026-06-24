; RUN: llc < %s -mtriple=sparc -mattr=fix-tn0010 | FileCheck %s

; CHECK-LABEL: empty_bb:
define i32 @empty_bb() {
  unreachable
}

; CHECK-LABEL: simulator_kernel:
define i32 @simulator_kernel() {
entry:
  %v = load i32, ptr null, align 4
  br label %dispatch

dispatch:
  br label %indirectgoto

store_block:
  store i32 %v, ptr null, align 4
  br label %indirectgoto

indirectgoto:
  indirectbr ptr null, [label %dispatch, label %store_block]
}
