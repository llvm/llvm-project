; RUN: llc < %s
; <rdar://problem/7499313>
target triple = "x86_64-apple-darwin8"

declare void @func2(i16 zeroext)

define void @func1() nounwind {
entry:
  %t1 = icmp ne i8 undef, 0
  %t2 = icmp eq i8 undef, 14
  %t3 = and i1 %t1, %t2
  %t4 = select i1 %t3, i16 0, i16 128
  call void @func2(i16 zeroext %t4) nounwind
  ret void
}
