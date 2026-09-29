; Suppress redundant loads feeding into fake uses.
; RUN: llc -filetype=asm -o - %s --mtriple=x86_64-unknown-unknown | FileCheck %s
; Windows ABI works differently, there's no offset.
;
; Look for the spill
; CHECK:      movq %r{{[a-z]+,}} -{{[0-9]+\(%rsp\)}}
; CHECK-NOT:  movq -{{[0-9]+\(%rsp\)}}, %r{{[a-z]+}}

define dso_local i32 @f(ptr %p) local_unnamed_addr optdebug {
entry:
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"() #1
  notail call void (...) @llvm.fake.use(ptr %p)
  ret i32 4
}
