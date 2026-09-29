; RUN: llc < %s -O1 -mtriple=i386-apple-darwin -x86-asm-syntax=intel | FileCheck %s
;
; Interesting test case where %tmp1220 = xor i32 %tmp862, %tmp592 and
; %tmp1676 = xor i32 %tmp1634, %tmp1530 have zero demanded bits after
; DAGCombiner optimization pass.  These are changed to undef and in turn
; the successor shl(s) become shl undef, 1.  This pattern then matches
; shl x, 1 -> add x, x.  add undef, undef doesn't guarantee the low
; order bit is zero and is incorrect.
;
; See rdar://9453156 and rdar://9487392.
;

; Use intel syntax, or "shl" might hit "pushl".

; CHECK-NOT: shl
define i32 @foo(ptr %a0, ptr %a2) nounwind {
entry:
  %tmp0 = alloca i8
  %tmp1 = alloca i32
  store i8 1, ptr %tmp0
  %tmp921.i7845 = load i8, ptr %a0, align 1
  %tmp309 = xor i8 %tmp921.i7845, 104
  %tmp592 = zext i8 %tmp309 to i32
  %tmp862 = xor i32 1293461297, %tmp592
  %tmp1220 = xor i32 %tmp862, %tmp592
  %tmp1506 = shl i32 %tmp1220, 1
  %tmp1530 = sub i32 %tmp592, %tmp1506
  %tmp1557 = sub i32 %tmp1530, 542767629
  %tmp1607 = and i32 %tmp1557, 1
  store i32 %tmp1607, ptr %tmp1
  %tmp1634 = and i32 %tmp1607, 2080309246
  %tmp1676 = xor i32 %tmp1634, %tmp1530
  %tmp1618 = shl i32 %tmp1676, 1
  %tmp1645 = sub i32 %tmp862, %tmp1618
  %tmp1697 = and i32 %tmp1645, 1
  store i32 %tmp1697, ptr %a2
  ret i32 %tmp1607
}

; CHECK-NOT: shl
; shl undef, 0 -> undef
define i32 @foo2_undef() nounwind {
entry:
  %tmp2 = shl i32 undef, 0;
  ret i32 %tmp2
}

; CHECK-NOT: shl
; shl undef, x -> 0
define i32 @foo1_undef(ptr %a0) nounwind {
entry:
  %tmp1 = load i32, ptr %a0, align 1
  %tmp2 = shl i32 undef, %tmp1;
  ret i32 %tmp2
}
