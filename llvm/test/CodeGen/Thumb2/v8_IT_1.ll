; RUN: llc < %s -mtriple=thumbv8 -mattr=+neon | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7 -mattr=+neon -arm-restrict-it | FileCheck %s

;CHECK-LABEL: select_s_v_v:
;CHECK-NOT: it
;CHECK: bx
define <16 x i8> @select_s_v_v(i32 %avail, ptr %bar) {
entry:
  %vld1 = call <16 x i8> @llvm.arm.neon.vld1.v16i8.p0(ptr %bar, i32 1)
  %and = and i32 %avail, 1
  %tobool = icmp eq i32 %and, 0
  %vld1. = select i1 %tobool, <16 x i8> %vld1, <16 x i8> zeroinitializer
  ret <16 x i8> %vld1.
}

declare <16 x i8> @llvm.arm.neon.vld1.v16i8.p0(ptr , i32 )

