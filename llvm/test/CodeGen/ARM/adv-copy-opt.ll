; RUN: llc -O1 -mtriple=armv7s-apple-ios -mcpu=swift < %s -disable-adv-copy-opt=true | FileCheck -check-prefix=NOOPT --check-prefix=CHECK %s
; RUN: llc -O1 -mtriple=armv7s-apple-ios -mcpu=swift < %s -disable-adv-copy-opt=false | FileCheck -check-prefix=OPT --check-prefix=CHECK %s
; RUN: llc -O1 -mtriple=thumbv7s-apple-ios -mcpu=swift < %s -disable-adv-copy-opt=true | FileCheck -check-prefix=NOOPT --check-prefix=CHECK %s
; RUN: llc -O1 -mtriple=thumbv7s-apple-ios -mcpu=swift < %s -disable-adv-copy-opt=false | FileCheck -check-prefix=OPT --check-prefix=CHECK %s

; CHECK-LABEL: simpleVectorDiv
; ABI: %A => r0, r1.
;      %B => r2, r3
;      ret => r0, r1
; We want to compute:
; r0 = r0 / r2
; r1 = r1 / r3
;
; NOOPT: vmov	[[A:d[0-9]+]], r0, r1
; NOOPT-NEXT: vmov	[[B:d[0-9]+]], r2, r3
; Move the low part of B into a register.
; Unfortunately, we cannot express that the 's' register is the low
; part of B, i.e., sIdx == BIdx x 2. E.g., B = d1, B_low = s2.
; NOOPT-NEXT: vmov	[[B_LOW:r[0-9]+]], s{{[0-9]+}}
; NOOPT-NEXT: vmov	[[B_HIGH:r[0-9]+]], s{{[0-9]+}}
; NOOPT-NEXT: vmov	[[A_LOW:r[0-9]+]], s{{[0-9]+}}
; NOOPT-NEXT: vmov	[[A_HIGH:r[0-9]+]], s{{[0-9]+}}
; NOOPT-NEXT: udiv	[[RES_LOW:r[0-9]+]], [[A_LOW]], [[B_LOW]]
; NOOPT-NEXT: vmov.32	[[RES:d[0-9]+]][0], [[RES_LOW]]
; NOOPT-NEXT: udiv	[[RES_HIGH:r[0-9]+]], [[A_HIGH]], [[B_HIGH]]
; NOOPT-NEXT: vmov.32	[[RES]][1], [[RES_HIGH]]
; NOOPT-NEXT: vmov	r0, r1, [[RES]]
; NOOPT-NEXT: bx	lr
;
; OPT-NOT: vmov
; OPT: udiv	r1, r1, r3
; OPT-NEXT: 	udiv	r0, r0, r2
; OPT-NEXT: bx	lr
define <2 x i32> @simpleVectorDiv(<2 x i32> %A, <2 x i32> %B) nounwind {
entry:
  %div = udiv <2 x i32> %A, %B
  ret <2 x i32> %div
}
