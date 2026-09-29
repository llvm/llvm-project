; RUN: opt -passes='print<structural-hash>' -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes='print<structural-hash><detailed>' -disable-output %s 2>&1 | FileCheck %s -check-prefix=DETAILED-HASH
; RUN: opt -passes='print<structural-hash><call-target-ignored>' -disable-output %s 2>&1 | FileCheck %s -check-prefix=CALLTARGETIGNORED-HASH

; Add a declaration so that we can test we skip it.
declare i64 @d1(i64)
declare i64 @e1(i64)

define i64 @f1(i64 %a) {
	%b = add i64 %a, 1
	%c = call i64 @d1(i64 %b)
	ret i64 %c
}

define i64 @f2(i64 %a) {
	%b = add i64 %a, 1
	%c = call i64 @e1(i64 %b)
	ret i64 %c
}

; CHECK: Module Hash: {{([a-f0-9]{16,})}}
; CHECK-NEXT: Function f1 Hash: [[F1H:([a-f0-9]{16,})]]
; CHECK-NEXT: Function f2 Hash: [[F1H]]

; DETAILED-HASH: Module Hash: {{([a-f0-9]{16,})}}
; DETAILED-HASH-NEXT: Function f1 Hash: [[DF1H:([a-f0-9]{16,})]]
; DETAILED-HASH-NOT: [[DF1H]]
; DETAILED-HASH-NEXT: Function f2 Hash: {{([a-f0-9]{16,})}}

; When ignoring the call target, check if `f1` and `f2` produce the same function hash.
; The index for the call instruction is 1, and the index of the call target operand is 1.
; The ignored operand hashes for different call targets should be different.
; CALLTARGETIGNORED-HASH: Module Hash: {{([a-f0-9]{16,})}}
; CALLTARGETIGNORED-HASH-NEXT: Function f1 Hash: [[IF1H:([a-f0-9]{16,})]]
; CALLTARGETIGNORED-HASH-NEXT:   Ignored Operand Hash: [[IO1H:([a-f0-9]{16,})]] at (1,1)
; CALLTARGETIGNORED-HASH-NEXT: Function f2 Hash: [[IF1H]]
; CALLTARGETIGNORED-HASH-NOT: [[IO1H]]
; CALLTARGETIGNORED-HASH-NEXT:   Ignored Operand Hash: {{([a-f0-9]{16,})}} at (1,1)
