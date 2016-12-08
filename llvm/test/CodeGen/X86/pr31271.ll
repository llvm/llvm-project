; RUN: llc -mtriple=i386-unknown-linux-gnu < %s | FileCheck %s

@c = external global [1 x i32], align 4

; CHECK-LABEL: fn1
; CHECK: leal c(%eax), %ecx
define void @fn1(i32 %k) {
  %g = getelementptr inbounds [1 x i32], [1 x i32]* @c, i32 0, i32 %k
  %cmp = icmp ne i32* undef, %g
  %z = zext i1 %cmp to i32
  store i32 %z, i32* undef, align 4
  %cmp2 = icmp eq i32* %g, null
  br i1 %cmp2, label %u, label %r

u:
  unreachable

r:
  ret void
}
