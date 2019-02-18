; RUN: opt -simple-loop-unswitch -S < %s | FileCheck %s
; RUN: opt -simple-loop-unswitch -enable-mssa-loop-dependency=true -verify-memoryssa -S < %s | FileCheck %s

define void @f(i32 %n, i32* %ptr) {
; CHECK-LABEL: @f(
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %be ]
  %iv.inc = add i32 %iv, 1
  %unswitch_cond_root = icmp ne i32 %iv.inc, 42
  %us.0 = and i1 %unswitch_cond_root, %unswitch_cond_root
  %us.1 = and i1 %us.0, %us.0
  %us.2 = and i1 %us.1, %us.1
  %us.3 = and i1 %us.2, %us.2
  %us.4 = and i1 %us.3, %us.3
  %us.5 = and i1 %us.4, %us.4
  %us.6 = and i1 %us.5, %us.5
  %us.7 = and i1 %us.6, %us.6
  %us.8 = and i1 %us.7, %us.7
  %us.9 = and i1 %us.8, %us.8
  %us.10 = and i1 %us.9, %us.9
  %us.11 = and i1 %us.10, %us.10
  %us.12 = and i1 %us.11, %us.11
  %us.13 = and i1 %us.12, %us.12
  %us.14 = and i1 %us.13, %us.13
  %us.15 = and i1 %us.14, %us.14
  %us.16 = and i1 %us.15, %us.15
  %us.17 = and i1 %us.16, %us.16
  %us.18 = and i1 %us.17, %us.17
  %us.19 = and i1 %us.18, %us.18
  %us.20 = and i1 %us.19, %us.19
  %us.21 = and i1 %us.20, %us.20
  %us.22 = and i1 %us.21, %us.21
  %us.23 = and i1 %us.22, %us.22
  %us.24 = and i1 %us.23, %us.23
  %us.25 = and i1 %us.24, %us.24
  %us.26 = and i1 %us.25, %us.25
  %us.27 = and i1 %us.26, %us.26
  %us.28 = and i1 %us.27, %us.27
  %us.29 = and i1 %us.28, %us.28
  br i1 %us.29, label %leave, label %be

be:
  store volatile i32 0, i32* %ptr
  %becond = icmp ult i32 %iv.inc, %n
  br i1 %becond, label %leave, label %loop

leave:
  ret void
}
