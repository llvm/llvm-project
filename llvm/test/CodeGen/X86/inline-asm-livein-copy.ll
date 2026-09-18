; RUN: llc -mtriple=x86_64-linux-gnu -O2 -frame-pointer=all < %s | FileCheck %s

@counter = internal global i64 0
@counter.1 = internal global i64 0

define void @livein_copy(i64 %arg) {
; CHECK-LABEL: livein_copy:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pushq %rbp
; CHECK:         movq %rdi, {{[-0-9]+}}(%rbp) # 8-byte Spill
; CHECK:         #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK:         movq {{[-0-9]+}}(%rbp), %{{[a-z0-9]+}} # 8-byte Reload
; CHECK:         #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK:         retq
  %count = load i64, ptr @counter
  %inc = add i64 %count, 1
  store i64 %inc, ptr @counter
  %count.1 = load i64, ptr @counter.1
  %inc.1 = add i64 %count.1, 1
  store i64 %inc.1, ptr @counter.1
  tail call { i64, i64, i64 } asm sideeffect "nop",
      "=r,=r,=r,0,1,2,~{rax},~{rbx},~{rcx},~{rdx},~{r8},~{r9},~{r10},~{r11},~{r13},~{r14},~{r15},~{memory},~{cc}"
      (i64 4660, i64 43981, i64 239)
  %count.2 = load i64, ptr @counter.1
  %inc.2 = add i64 %count.2, 1
  store i64 %inc.2, ptr @counter.1
  tail call { i64, i64, i64 } asm sideeffect "nop",
      "=r,=r,=r,0,1,2,~{rax},~{rbx},~{rcx},~{rdx},~{r8},~{r9},~{r10},~{r11},~{r13},~{r14},~{r15},~{memory},~{cc}"
      (i64 6699, i64 15437, i64 %arg)
  ret void
}
