; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 -mattr=+vfp2 %s -o - \
; RUN:  | FileCheck %s

; rdar://7461510
; rdar://10964603

; Disable this optimization unless we know one of them is zero.
define arm_apcscc i32 @t1(ptr %a, ptr %b) nounwind {
entry:
; CHECK-LABEL: t1:
; CHECK: vldr [[S0:s[0-9]+]],
; CHECK: vldr [[S1:s[0-9]+]],
; CHECK: vcmp.f32 [[S1]], [[S0]]
; CHECK: vmrs APSR_nzcv, fpscr
; CHECK: beq
  %0 = load float, ptr %a
  %1 = load float, ptr %b
  %2 = fcmp une float %0, %1
  br i1 %2, label %bb1, label %bb2

bb1:
  %3 = call i32 @bar()
  ret i32 %3

bb2:
  %4 = call i32 @foo()
  ret i32 %4
}

; If one side is zero, the other size sign bit is masked off to allow
; +0.0 == -0.0
define arm_apcscc i32 @t2(ptr %a, ptr %b) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK-NOT: vldr
; CHECK: ldrd [[REG1:(r[0-9]+)]], [[REG2:(r[0-9]+)]], [r0]
; CHECK-NOT: b LBB
; CHECK: bic [[REG2]], [[REG2]], #-2147483648
; CHECK: cmp [[REG1]], #0
; CHECK: cmpeq [[REG2]], #0
; CHECK-NOT: vcmp.f32
; CHECK-NOT: vmrs
; CHECK: bne
  %0 = load double, ptr %a
  %1 = fcmp nnan oeq double %0, 0.000000e+00
  br i1 %1, label %bb1, label %bb2

bb1:
  %2 = call i32 @bar()
  ret i32 %2

bb2:
  %3 = call i32 @foo()
  ret i32 %3
}

define arm_apcscc i32 @t3(ptr %a, ptr %b) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK-NOT: vldr
; CHECK: ldr [[REG3:(r[0-9]+)]], [r0]
; CHECK: mvn [[REG4:(r[0-9]+)]], #-2147483648
; CHECK: tst [[REG3]], [[REG4]]
; CHECK-NOT: vcmp.f32
; CHECK-NOT: vmrs
; CHECK: bne
  %0 = load float, ptr %a
  %1 = fcmp nnan oeq float %0, 0.000000e+00
  br i1 %1, label %bb1, label %bb2

bb1:
  %2 = call i32 @bar()
  ret i32 %2

bb2:
  %3 = call i32 @foo()
  ret i32 %3
}

declare i32 @bar()
declare i32 @foo()
