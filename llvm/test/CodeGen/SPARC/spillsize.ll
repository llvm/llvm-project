; RUN: llc < %s -verify-machineinstrs | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64-S128"
target triple = "sparcv9"

; CHECK-LABEL: spill4
; This function spills two values: %p and the materialized large constant.
; Both must use 8-byte spill and fill instructions.
; CHECK: stx %{{..}}, [%fp+
; CHECK: stx %{{..}}, [%fp+
; CHECK: ldx [%fp+
; CHECK: ldx [%fp+
define void @spill4(i64* nocapture %p) {
entry:
  %val0 = load i64, i64* %p
  %cmp0 = icmp ult i64 %val0, 385672958347594845
  %cm80 = zext i1 %cmp0 to i64
  store i64 %cm80, i64* %p, align 8
  tail call void asm sideeffect "", "~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{g2},~{g3},~{g4},~{g5},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o7}"()
  %arrayidx1 = getelementptr inbounds i64, i64* %p, i64 1
  %val = load i64, i64* %arrayidx1
  %cmp = icmp ult i64 %val, 385672958347594845
  %cm8 = select i1 %cmp, i64 10, i64 20
  store i64 %cm8, i64* %arrayidx1, align 8
  ret void
}
