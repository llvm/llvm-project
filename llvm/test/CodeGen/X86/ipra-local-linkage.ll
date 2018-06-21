; RUN: llc < %s | FileCheck %s -check-prefix=NOIPRA
; RUN: llc -enable-ipra < %s | FileCheck %s

target triple = "x86_64--"

define internal void @foo() norecurse {
; When IPRA is not enabled R15 will be saved by foo as it is callee saved reg.
; NOIPRA-LABEL: foo:
; NOIPRA: pushq	%r15
; When IPRA is enabled none register should be saved as foo() is local function
; so we optimize it to save no registers.
; CHECK-LABEL: foo:
; CHECK-NOT: pushq %r15
  call void asm sideeffect "movl	%r14d, %r15d", "~{r15}"()
  ret void
}

define void @bar(i32 %X) {
  call void asm sideeffect "movl  %r12d, $0", "{r15}~{r12}"(i32 %X)
  ; As R15 is clobbered by foo() when IPRA is enabled value of R15 should be
  ; saved if register containing orignal value is also getting clobbered
  ; and reloaded after foo(), here original value is loaded back into R15D after
  ; call to foo.
  call void @foo()
  ; CHECK-LABEL: bar:
  ; CHECK: callq foo
  ; CHECK-NEXT: movl  %eax, %r15d
  call void asm sideeffect "movl  $0, %r12d", "{r15}~{r12}"(i32 %X)
  ret void
}
