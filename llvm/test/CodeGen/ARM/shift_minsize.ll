; RUN: llc -mtriple=arm-eabi        %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7-windows %s -o - | FileCheck %s -check-prefix=CHECK-WIN

; The Windows runtime doesn't have these.
; CHECK-WIN-NOT: __ashldi3
; CHECK-WIN-NOT: __ashrdi3
; CHECK-WIN-NOT: __lshrdi3

define i64 @f0(i64 %val, i64 %amt) minsize optsize {
; CHECK-LABEL:   f0:
; CHECK:         bl  __aeabi_llsl
  %res = shl i64 %val, %amt
  ret i64 %res
}

define i32 @f1(i64 %x, i64 %y) minsize optsize {
; CHECK-LABEL:   f1:
; CHECK:         bl  __aeabi_llsl
	%a = shl i64 %x, %y
	%b = trunc i64 %a to i32
	ret i32 %b
}

define i32 @f2(i64 %x, i64 %y) minsize optsize {
; CHECK-LABEL:   f2:
; CHECK:         bl  __aeabi_lasr
	%a = ashr i64 %x, %y
	%b = trunc i64 %a to i32
	ret i32 %b
}

define i32 @f3(i64 %x, i64 %y) minsize optsize {
; CHECK-LABEL:   f3:
; CHECK:         bl  __aeabi_llsr
	%a = lshr i64 %x, %y
	%b = trunc i64 %a to i32
	ret i32 %b
}
