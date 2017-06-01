; RUN: opt -S -guard-widening < %s | FileCheck %s

declare void @llvm.experimental.guard(i1,...)

define void @f_0(i32 %x, i32* %length_buf) {
; CHECK-LABEL: @f_0(
; CHECK-NOT: @llvm.experimental.guard
; CHECK:  %wide.chk2 = and i1 %chk3, %chk0
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk2) [ "deopt"() ]
; CHECK:  ret void
entry:
  %length = load i32, i32* %length_buf, !range !0
  %chk0 = icmp ult i32 %x, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk0) [ "deopt"() ]

  %x.inc1 = add i32 %x, 1
  %chk1 = icmp ult i32 %x.inc1, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk1) [ "deopt"() ]

  %x.inc2 = add i32 %x, 2
  %chk2 = icmp ult i32 %x.inc2, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk2) [ "deopt"() ]

  %x.inc3 = add i32 %x, 3
  %chk3 = icmp ult i32 %x.inc3, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk3) [ "deopt"() ]
  ret void
}

define void @f_1(i32 %x, i32* %length_buf) {
; CHECK-LABEL: @f_1(
; CHECK-NOT: llvm.experimental.guard
; CHECK:  %wide.chk2 = and i1 %chk3, %chk0
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk2) [ "deopt"() ]
; CHECK:  ret void
entry:
  %length = load i32, i32* %length_buf, !range !0
  %chk0 = icmp ult i32 %x, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk0) [ "deopt"() ]

  %x.inc1 = add i32 %x, 1
  %chk1 = icmp ult i32 %x.inc1, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk1) [ "deopt"() ]

  %x.inc2 = add i32 %x.inc1, 2
  %chk2 = icmp ult i32 %x.inc2, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk2) [ "deopt"() ]

  %x.inc3 = add i32 %x.inc2, 3
  %chk3 = icmp ult i32 %x.inc3, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk3) [ "deopt"() ]
  ret void
}

define void @f_2(i32 %a, i32* %length_buf) {
; CHECK-LABEL: @f_2(
; CHECK-NOT: llvm.experimental.guard
; CHECK:  %wide.chk2 = and i1 %chk3, %chk0
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk2) [ "deopt"() ]
; CHECK:  ret void
entry:
  %x = and i32 %a, 4294967040 ;; 4294967040 == 0xffffff00
  %length = load i32, i32* %length_buf, !range !0
  %chk0 = icmp ult i32 %x, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk0) [ "deopt"() ]

  %x.inc1 = or i32 %x, 1
  %chk1 = icmp ult i32 %x.inc1, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk1) [ "deopt"() ]

  %x.inc2 = or i32 %x, 2
  %chk2 = icmp ult i32 %x.inc2, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk2) [ "deopt"() ]

  %x.inc3 = or i32 %x, 3
  %chk3 = icmp ult i32 %x.inc3, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk3) [ "deopt"() ]
  ret void
}

define void @f_3(i32 %a, i32* %length_buf) {
; CHECK-LABEL: @f_3(
; CHECK-NOT: llvm.experimental.guard
; CHECK:  %wide.chk2 = and i1 %chk3, %chk0
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk2) [ "deopt"() ]
; CHECK:  ret void
entry:
  %x = and i32 %a, 4294967040 ;; 4294967040 == 0xffffff00
  %length = load i32, i32* %length_buf, !range !0
  %chk0 = icmp ult i32 %x, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk0) [ "deopt"() ]

  %x.inc1 = add i32 %x, 1
  %chk1 = icmp ult i32 %x.inc1, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk1) [ "deopt"() ]

  %x.inc2 = or i32 %x.inc1, 2
  %chk2 = icmp ult i32 %x.inc2, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk2) [ "deopt"() ]

  %x.inc3 = add i32 %x.inc2, 3
  %chk3 = icmp ult i32 %x.inc3, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk3) [ "deopt"() ]
  ret void
}

define void @f_4(i32 %x, i32* %length_buf) {
; CHECK-LABEL: @f_4(
; CHECK-NOT: llvm.experimental.guard

; Note: we NOT guarding on "and i1 %chk3, %chk0", that would be incorrect.
; CHECK:  %wide.chk2 = and i1 %chk3, %chk1
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk2) [ "deopt"() ]
; CHECK:  ret void
entry:
  %length = load i32, i32* %length_buf, !range !0
  %chk0 = icmp ult i32 %x, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk0) [ "deopt"() ]

  %x.inc1 = add i32 %x, -1024
  %chk1 = icmp ult i32 %x.inc1, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk1) [ "deopt"() ]

  %x.inc2 = add i32 %x, 2
  %chk2 = icmp ult i32 %x.inc2, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk2) [ "deopt"() ]

  %x.inc3 = add i32 %x, 3
  %chk3 = icmp ult i32 %x.inc3, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk3) [ "deopt"() ]
  ret void
}

define void @f_5(i32 %x, i32* %length_buf) {
; CHECK-LABEL: @f_5(
; CHECK-NOT: llvm.experimental.guard
; CHECK:  %wide.chk2 = and i1 %chk1, %chk2
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk2) [ "deopt"() ]
; CHECK:  ret void
entry:
  %length = load i32, i32* %length_buf, !range !0
  %chk0 = icmp ult i32 %x, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk0) [ "deopt"() ]

  %x.inc1 = add i32 %x, 1
  %chk1 = icmp ult i32 %x.inc1, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk1) [ "deopt"() ]

  %x.inc2 = add i32 %x.inc1, -200
  %chk2 = icmp ult i32 %x.inc2, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk2) [ "deopt"() ]

  %x.inc3 = add i32 %x.inc2, 3
  %chk3 = icmp ult i32 %x.inc3, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk3) [ "deopt"() ]
  ret void
}


; Negative test: we can't merge these checks into
;
;  (%x + -2147483647) u< L && (%x + 3) u< L
;
; because if %length == INT_MAX and %x == -3 then
;
; (%x + -2147483647) == i32 2147483646  u< L   (L is 2147483647)
; (%x + 3) == 0 u< L
;
; But (%x + 2) == -1 is not u< L
;
define void @f_6(i32 %x, i32* %length_buf) {
; CHECK-LABEL: @f_6(
; CHECK-NOT: llvm.experimental.guard
; CHECK:  %wide.chk = and i1 %chk0, %chk1
; CHECK:  %wide.chk1 = and i1 %wide.chk, %chk2
; CHECK:  %wide.chk2 = and i1 %wide.chk1, %chk3
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk2) [ "deopt"() ]
entry:
  %length = load i32, i32* %length_buf, !range !0
  %chk0 = icmp ult i32 %x, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk0) [ "deopt"() ]

  %x.inc1 = add i32 %x, -2147483647 ;; -2147483647 == (i32 INT_MIN)+1 == -(i32 INT_MAX)
  %chk1 = icmp ult i32 %x.inc1, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk1) [ "deopt"() ]

  %x.inc2 = add i32 %x, 2
  %chk2 = icmp ult i32 %x.inc2, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk2) [ "deopt"() ]

  %x.inc3 = add i32 %x, 3
  %chk3 = icmp ult i32 %x.inc3, %length
  call void(i1, ...) @llvm.experimental.guard(i1 %chk3) [ "deopt"() ]
  ret void
}


define void @f_7(i32 %x, i32* %length_buf) {
; CHECK-LABEL: @f_7(

; CHECK:  [[COND_0:%[^ ]+]] = and i1 %chk3.b, %chk0.b
; CHECK:  [[COND_1:%[^ ]+]] = and i1 %chk0.a, [[COND_0]]
; CHECK:  [[COND_2:%[^ ]+]] = and i1 %chk3.a, [[COND_1]]
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 [[COND_2]]) [ "deopt"() ]

entry:
  %length_a = load volatile i32, i32* %length_buf, !range !0
  %length_b = load volatile i32, i32* %length_buf, !range !0
  %chk0.a = icmp ult i32 %x, %length_a
  %chk0.b = icmp ult i32 %x, %length_b
  %chk0 = and i1 %chk0.a, %chk0.b
  call void(i1, ...) @llvm.experimental.guard(i1 %chk0) [ "deopt"() ]

  %x.inc1 = add i32 %x, 1
  %chk1.a = icmp ult i32 %x.inc1, %length_a
  %chk1.b = icmp ult i32 %x.inc1, %length_b
  %chk1 = and i1 %chk1.a, %chk1.b
  call void(i1, ...) @llvm.experimental.guard(i1 %chk1) [ "deopt"() ]

  %x.inc2 = add i32 %x, 2
  %chk2.a = icmp ult i32 %x.inc2, %length_a
  %chk2.b = icmp ult i32 %x.inc2, %length_b
  %chk2 = and i1 %chk2.a, %chk2.b
  call void(i1, ...) @llvm.experimental.guard(i1 %chk2) [ "deopt"() ]

  %x.inc3 = add i32 %x, 3
  %chk3.a = icmp ult i32 %x.inc3, %length_a
  %chk3.b = icmp ult i32 %x.inc3, %length_b
  %chk3 = and i1 %chk3.a, %chk3.b
  call void(i1, ...) @llvm.experimental.guard(i1 %chk3) [ "deopt"() ]
  ret void
}


!0 = !{i32 0, i32 2147483648}
