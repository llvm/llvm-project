; Checks that the yk-linkage pass changes functions to have external linkage.
;
; RUN: llc --yk-linkage -o - < %s | FileCheck %s
; RUN: llc -o - < %s | FileCheck --check-prefix CHECK-NOPASS %s

; CHECK: .globl myfunc
; CHECK-NOPASS-NOT: .globl myfunc
define internal void @myfunc() noinline {
	ret void
}
