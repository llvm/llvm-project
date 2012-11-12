; RUN: llc < %s -mtriple x86_64-apple-darwin -O0 | FileCheck %s

@b = thread_local global i32 5, align 4
@a = thread_local global i32 0, align 4
@c = internal thread_local global i32 0, align 4
@d = internal thread_local global i32 5, align 4

define void @foo() nounwind ssp {
entry:
  store i32 1, i32* @a, align 4
  ; CHECK: movq    _a@TLVP(%rip), %rdi
  ; CHECK: callq   *(%rdi)
  ; CHECK: movl    $1, (%rax)
  
  store i32 2, i32* @b, align 4
  ; CHECK: movq    _b@TLVP(%rip), %rdi
  ; CHECK: callq   *(%rdi)
  ; CHECK: movl    $2, (%rax)

  store i32 3, i32* @c, align 4
  ; CHECK: movq    _c@TLVP(%rip), %rdi
  ; CHECK: callq   *(%rdi)
  ; CHECK: movl    $3, (%rax)
  
  store i32 4, i32* @d, align 4
  ; CHECK: movq    _d@TLVP(%rip), %rdi
  ; CHECK: callq   *(%rdi)
  ; CHECK: movl    $4, (%rax)
  ; CHECK: popq
  
  ret void
}
