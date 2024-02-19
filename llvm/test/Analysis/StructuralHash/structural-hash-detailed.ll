; Ensure that the value of the structural hash is consistent across runs to
; check that we aren't doing something like hashing a pointer that could
; introduce non-determinism.

; RUN: opt -passes='print<structural-hash><detailed>' -disable-output %s &> %t.1
; RUN: opt -passes='print<structural-hash><detailed>' -disable-output %s &> %t.2
; RUN: diff %t.1 %t.2

; Check that we get valid output in the detailed case.

; cat %t.1 | FileCheck %s

define i64 @f1(i64 %a) {
	ret i64 %a
}

; CHECK: Module Hash: {{([a-z0-9]{14,})}}
; CHECK: Function f1 Hash: {{([a-z0-9]{14,})}}

