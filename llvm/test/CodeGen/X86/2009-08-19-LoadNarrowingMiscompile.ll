; RUN: llc < %s -march=x86 -mtriple=i386-pc-linux | FileCheck %s

@a = external global i96, align 4
@b = external global i64, align 8

define void @c() nounwind {
; CHECK: movl a+8, %eax
  %srcval1 = load i96* @a, align 4
  %sroa.store.elt2 = lshr i96 %srcval1, 64
  %tmp = trunc i96 %sroa.store.elt2 to i64
; CHECK: movl %eax, b
; CHECK: movl $0, b+4
  store i64 %tmp, i64* @b, align 8
  ret void
}
