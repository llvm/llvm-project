; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux | FileCheck %s

@c = external local_unnamed_addr global i32, align 4
@b = external local_unnamed_addr global i32, align 4
@d = external local_unnamed_addr global i32, align 4

; CHECK: cmpl    $1, c(%rip)
; CHECK-NEXT: sbbl    %eax, %eax
; CHECK-NEXT: andl    $1, %eax
; CHECK-NEXT: movl    %eax, d(%rip)
; CHECK-NEXT: retq

define void @fn1() {
entry:
  %0 = load i32, i32* @c, align 4
  %tobool1 = icmp eq i32 %0, 0
  %xor = zext i1 %tobool1 to i32
  %1 = load i32, i32* @b, align 4
  %tobool2 = icmp ne i32 %1, 0
  %tobool4 = icmp ne i32 undef, 0
  %2 = and i1 %tobool4, %tobool2
  %sub = sext i1 %2 to i32
  %div = sdiv i32 %sub, 2
  %add = add nsw i32 %div, %xor
  store i32 %add, i32* @d, align 4
  ret void
}
