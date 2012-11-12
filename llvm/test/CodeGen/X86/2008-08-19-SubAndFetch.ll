; RUN: llc < %s -march=x86-64 | FileCheck %s

@var = external global i64		; <i64*> [#uses=1]

define i32 @main() nounwind {
entry:
; CHECK: main:
; CHECK: lock
; CHECK: decq
	atomicrmw sub i64* @var, i64 1 monotonic
	unreachable
}
