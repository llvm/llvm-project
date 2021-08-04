; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s

declare swifttailcc void @callee([6 x i64], i64, i64)

@var = external global i8*

define swifttailcc void @caller(i64 %n) {
; CHECK-LABEL: caller:
; CHECK: subq $16, %rsp
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: pushq %rbx
; CHECK: andq $-32, %rsp
; [... don't really care what happens to rsp to allocate %ptr ...]
; CHECK: movq 24(%rbp), [[RETADDR:%.*]]
; CHECK: movq [[RETADDR]], 8(%rbp)
; CHECK: movq $42, 16(%rbp)
; CHECK: movq $0, 24(%rbp)
; CHECK: leaq -8(%rbp), %rsp
; CHECK: popq %rbx
; CHECK: popq %rbp
; CHECK: jmp callee

  call void asm sideeffect "", "~{rbx}"()
  %ptr = alloca i8, i64 %n, align 32
  store i8* %ptr, i8** @var
  tail call swifttailcc void @callee([6 x i64] undef, i64 42, i64 0)
  ret void
}
