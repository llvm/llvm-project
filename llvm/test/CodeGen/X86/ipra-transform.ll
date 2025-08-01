
; RUN: llc < %s | FileCheck %s -check-prefix=NOIPRA
; RUN: llc -enable-ipra < %s | FileCheck %s


target triple = "x86_64-unknown-unknown"
define void @bar1() {
	ret void
}
define preserve_allcc void @foo()#0 {
; Due to preserve_allcc foo() will save some registers at start of foo()
; prefix NOIPRA will verify that.
; NOIPRA-LABEL: foo:
; NOIPRA: pushq	%r10
; NOIPRA-NEXT: pushq %r9
; NOIPRA-NEXT: pushq %r8
; NOIPRA: callq bar1
; When IPRA is present above registers will not be saved and that is verified
; by prefix CHECK.
; CHECK: foo:
; CHECK-NOT: pushq %r10
; CHECK-NOT: pushq %r9
; CHECK-NOT: pushq %r8
; CHECK: callq bar1
	call void @bar1()
	call void @bar2()
	ret void
}
define void @bar2() {
	ret void
}

define preserve_nonecc void @foo2()#0 {
; Due to preserve_nonecc foo2() will save above registers no matter IPRA is
; present or not.
; NOIPRA-LABEL: foo2:
; NOIPRA-NOT: pushq %r10
; NOIPRA-NOT: pushq %r9
; NOIPRA-NOT: pushq %r8
; NOIPRA: callq bar1
; CHECK: foo2:
; CHECK-NOT: pushq %r10
; CHECK-NOT: pushq %r9
; CHECK-NOT: pushq %r8
; CHECK: callq bar1
	call void @bar1()
	call void @bar2()
	ret void
}

attributes #0 = {nounwind}
