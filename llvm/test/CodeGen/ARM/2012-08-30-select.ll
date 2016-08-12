; RUN: llc < %s -mtriple=thumbv7-apple-ios | FileCheck %s
; rdar://12201387

;CHECK-LABEL: select_s_v_v:
;CHECK: itee  ne
;CHECK-NEXT: vmovne.i32
;CHECK-NEXT: vmoveq
;CHECK-NEXT: vmoveq
;CHECK: bx
define <16 x i8> @select_s_v_v(<16 x i8> %vec, i32 %avail) {
entry:
  %and = and i32 %avail, 1
  %tobool = icmp eq i32 %and, 0
  %ret = select i1 %tobool, <16 x i8> %vec, <16 x i8> zeroinitializer
  ret <16 x i8> %ret
}

