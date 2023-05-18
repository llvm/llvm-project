; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-misched=false < %s | FileCheck %s

define i32 @fold64to32(i64 %add, i32 %spill) {
; CHECK-LABEL: fold64to32:
; CHECK:    movq %rdi, -{{[0-9]+}}(%rsp) # 8-byte Spill
; CHECK:    subl -{{[0-9]+}}(%rsp), %esi # 4-byte Folded Reload
entry:
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"()
  %trunc = trunc i64 %add to i32
  %sub = sub i32 %spill, %trunc
  ret i32 %sub
}

define i8 @fold64to8(i64 %add, i8 %spill) {
; CHECK-LABEL: fold64to8:
; CHECK:    movq %rdi, -{{[0-9]+}}(%rsp) # 8-byte Spill
; CHECK:    subb -{{[0-9]+}}(%rsp), %sil # 1-byte Folded Reload
entry:
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"()
  %trunc = trunc i64 %add to i8
  %sub = sub i8 %spill, %trunc
  ret i8 %sub
}

; Do not fold a 4-byte store into a 8-byte spill slot
; CHECK-LABEL: nofold
; CHECK:    movq %rsi, -{{[0-9]+}}(%rsp) # 8-byte Spill
; CHECK:    movq -{{[0-9]+}}(%rsp), %rax # 8-byte Reload
; CHECK:    subl %edi, %eax
; CHECK:    movq %rax, -{{[0-9]+}}(%rsp) # 8-byte Spill
; CHECK:    movq -{{[0-9]+}}(%rsp), %rax # 8-byte Reload
define i32 @nofold(i64 %add, i64 %spill) {
entry:
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"()
  %trunc = trunc i64 %add to i32
  %truncspill = trunc i64 %spill to i32
  %sub = sub i32 %truncspill, %trunc
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"()
  ret i32 %sub
}

