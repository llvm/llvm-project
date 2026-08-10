; RUN: llc -mtriple=thumbv6m-apple-macho %s -relocation-model=static -o - | FileCheck %s
; RUN: llc -mtriple=thumbv6m-apple-macho %s -relocation-model=pic -o - | FileCheck %s

@var = global i8 zeroinitializer

declare void @callee(ptr)

define void @foo() minsize {
; CHECK-LABEL: foo:
; CHECK: ldr {{r[0-7]}}, LCPI0_0
  call void @callee(ptr @var)
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7}"()
  call void @callee(ptr @var)
  ret void
}
