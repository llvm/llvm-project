; RUN: llc < %s -mtriple=aarch64-linux--gnu -aarch64-neon-syntax=generic | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: smax_B
; CHECK: smaxv {{b[0-9]+}}, {{v[0-9]+}}.16b
define i8 @smax_B(<16 x i8>* nocapture readonly %arr)  {
  %arr.load = load <16 x i8>, <16 x i8>* %arr
  %rdx.shuf = shufflevector <16 x i8> %arr.load, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp sgt <16 x i8> %arr.load, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i8> %arr.load, <16 x i8> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i8> %rdx.minmax.select23, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp sgt <16 x i8> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i8> %rdx.minmax.select23, <16 x i8> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i8> %rdx.minmax.select26, <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp sgt <16 x i8> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i8> %rdx.minmax.select26, <16 x i8> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i8> %rdx.minmax.select29, <16 x i8> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp sgt <16 x i8> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i8> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i8> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i8 %rdx.minmax.select29.elt, i8 %rdx.shuf30.elt
  ret i8 %r
}

; CHECK-LABEL: smax_H
; CHECK: smaxv {{h[0-9]+}}, {{v[0-9]+}}.8h
define i16 @smax_H(<8 x i16>* nocapture readonly %arr) {
  %rdx.minmax.select = load <8 x i16>, <8 x i16>* %arr
  %rdx.shuf = shufflevector <8 x i16> %rdx.minmax.select, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp23 = icmp sgt <8 x i16> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select24 = select <8 x i1> %rdx.minmax.cmp23, <8 x i16> %rdx.minmax.select, <8 x i16> %rdx.shuf
  %rdx.shuf25 = shufflevector <8 x i16> %rdx.minmax.select24, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp26 = icmp sgt <8 x i16> %rdx.minmax.select24, %rdx.shuf25
  %rdx.minmax.select27 = select <8 x i1> %rdx.minmax.cmp26, <8 x i16> %rdx.minmax.select24, <8 x i16> %rdx.shuf25
  %rdx.shuf28 = shufflevector <8 x i16> %rdx.minmax.select27, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp29 = icmp sgt <8 x i16> %rdx.minmax.select27, %rdx.shuf28
  %rdx.minmax.cmp29.elt = extractelement <8 x i1> %rdx.minmax.cmp29, i32 0
  %rdx.minmax.select27.elt = extractelement <8 x i16> %rdx.minmax.select27, i32 0
  %rdx.shuf28.elt = extractelement <8 x i16> %rdx.minmax.select27, i32 1
  %r = select i1 %rdx.minmax.cmp29.elt, i16 %rdx.minmax.select27.elt, i16 %rdx.shuf28.elt
  ret i16 %r
}

; CHECK-LABEL: smax_S
; CHECK: smaxv {{s[0-9]+}}, {{v[0-9]+}}.4s
define i32 @smax_S(<4 x i32> * nocapture readonly %arr)  {
  %rdx.minmax.select = load <4 x i32>, <4 x i32>* %arr
  %rdx.shuf = shufflevector <4 x i32> %rdx.minmax.select, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp18 = icmp sgt <4 x i32> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select19 = select <4 x i1> %rdx.minmax.cmp18, <4 x i32> %rdx.minmax.select, <4 x i32> %rdx.shuf
  %rdx.shuf20 = shufflevector <4 x i32> %rdx.minmax.select19, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp21 = icmp sgt <4 x i32> %rdx.minmax.select19, %rdx.shuf20
  %rdx.minmax.cmp21.elt = extractelement <4 x i1> %rdx.minmax.cmp21, i32 0
  %rdx.minmax.select19.elt = extractelement <4 x i32> %rdx.minmax.select19, i32 0
  %rdx.shuf20.elt = extractelement <4 x i32> %rdx.minmax.select19, i32 1
  %r = select i1 %rdx.minmax.cmp21.elt, i32 %rdx.minmax.select19.elt, i32 %rdx.shuf20.elt
  ret i32 %r
}

; CHECK-LABEL: smax_D
; CHECK-NOT: smaxv
define i64 @smax_D(<2 x i64>* nocapture readonly %arr) {
  %rdx.minmax.select = load <2 x i64>, <2 x i64>* %arr
  %rdx.shuf = shufflevector <2 x i64> %rdx.minmax.select, <2 x i64> undef, <2 x i32> <i32 1, i32 undef>
  %rdx.minmax.cmp18 = icmp sgt <2 x i64> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.cmp18.elt = extractelement <2 x i1> %rdx.minmax.cmp18, i32 0
  %rdx.minmax.select.elt = extractelement <2 x i64> %rdx.minmax.select, i32 0
  %rdx.shuf.elt = extractelement <2 x i64> %rdx.minmax.select, i32 1
  %r = select i1 %rdx.minmax.cmp18.elt, i64 %rdx.minmax.select.elt, i64 %rdx.shuf.elt
  ret i64 %r
}


; CHECK-LABEL: umax_B
; CHECK: umaxv {{b[0-9]+}}, {{v[0-9]+}}.16b
define i8 @umax_B(<16 x i8>* nocapture readonly %arr)  {
  %rdx.minmax.select = load <16 x i8>, <16 x i8>* %arr
  %rdx.shuf = shufflevector <16 x i8> %rdx.minmax.select, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp ugt <16 x i8> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i8> %rdx.minmax.select, <16 x i8> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i8> %rdx.minmax.select23, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp ugt <16 x i8> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i8> %rdx.minmax.select23, <16 x i8> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i8> %rdx.minmax.select26, <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp ugt <16 x i8> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i8> %rdx.minmax.select26, <16 x i8> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i8> %rdx.minmax.select29, <16 x i8> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp ugt <16 x i8> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i8> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i8> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i8 %rdx.minmax.select29.elt, i8 %rdx.shuf30.elt
  ret i8 %r
}

; CHECK-LABEL: umax_H
; CHECK: umaxv {{h[0-9]+}}, {{v[0-9]+}}.8h
define i16 @umax_H(<8 x i16>* nocapture readonly %arr)  {
  %rdx.minmax.select = load <8 x i16>, <8 x i16>* %arr
  %rdx.shuf = shufflevector <8 x i16> %rdx.minmax.select, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp23 = icmp ugt <8 x i16> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select24 = select <8 x i1> %rdx.minmax.cmp23, <8 x i16> %rdx.minmax.select, <8 x i16> %rdx.shuf
  %rdx.shuf25 = shufflevector <8 x i16> %rdx.minmax.select24, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp26 = icmp ugt <8 x i16> %rdx.minmax.select24, %rdx.shuf25
  %rdx.minmax.select27 = select <8 x i1> %rdx.minmax.cmp26, <8 x i16> %rdx.minmax.select24, <8 x i16> %rdx.shuf25
  %rdx.shuf28 = shufflevector <8 x i16> %rdx.minmax.select27, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp29 = icmp ugt <8 x i16> %rdx.minmax.select27, %rdx.shuf28
  %rdx.minmax.cmp29.elt = extractelement <8 x i1> %rdx.minmax.cmp29, i32 0
  %rdx.minmax.select27.elt = extractelement <8 x i16> %rdx.minmax.select27, i32 0
  %rdx.shuf28.elt = extractelement <8 x i16> %rdx.minmax.select27, i32 1
  %r = select i1 %rdx.minmax.cmp29.elt, i16 %rdx.minmax.select27.elt, i16 %rdx.shuf28.elt
  ret i16 %r
}

; CHECK-LABEL: umax_S
; CHECK: umaxv {{s[0-9]+}}, {{v[0-9]+}}.4s
define i32 @umax_S(<4 x i32>* nocapture readonly %arr) {
  %rdx.minmax.select  = load <4 x i32>, <4 x i32>* %arr
  %rdx.shuf = shufflevector <4 x i32> %rdx.minmax.select, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp18 = icmp ugt <4 x i32> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select19 = select <4 x i1> %rdx.minmax.cmp18, <4 x i32> %rdx.minmax.select, <4 x i32> %rdx.shuf
  %rdx.shuf20 = shufflevector <4 x i32> %rdx.minmax.select19, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp21 = icmp ugt <4 x i32> %rdx.minmax.select19, %rdx.shuf20
  %rdx.minmax.cmp21.elt = extractelement <4 x i1> %rdx.minmax.cmp21, i32 0
  %rdx.minmax.select19.elt = extractelement <4 x i32> %rdx.minmax.select19, i32 0
  %rdx.shuf20.elt = extractelement <4 x i32> %rdx.minmax.select19, i32 1
  %r = select i1 %rdx.minmax.cmp21.elt, i32 %rdx.minmax.select19.elt, i32 %rdx.shuf20.elt
  ret i32 %r
}

; CHECK-LABEL: umax_D
; CHECK-NOT: umaxv
define i64 @umax_D(<2 x i64>* nocapture readonly %arr)  {
  %rdx.minmax.select = load <2 x i64>, <2 x i64>* %arr
  %rdx.shuf = shufflevector <2 x i64> %rdx.minmax.select, <2 x i64> undef, <2 x i32> <i32 1, i32 undef>
  %rdx.minmax.cmp18 = icmp ugt <2 x i64> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.cmp18.elt = extractelement <2 x i1> %rdx.minmax.cmp18, i32 0
  %rdx.minmax.select.elt = extractelement <2 x i64> %rdx.minmax.select, i32 0
  %rdx.shuf.elt = extractelement <2 x i64> %rdx.minmax.select, i32 1
  %r = select i1 %rdx.minmax.cmp18.elt, i64 %rdx.minmax.select.elt, i64 %rdx.shuf.elt
  ret i64 %r
}


; CHECK-LABEL: smin_B
; CHECK: sminv {{b[0-9]+}}, {{v[0-9]+}}.16b
define i8 @smin_B(<16 x i8>* nocapture readonly %arr) {
  %rdx.minmax.select = load <16 x i8>, <16 x i8>* %arr
  %rdx.shuf = shufflevector <16 x i8> %rdx.minmax.select, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp slt <16 x i8> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i8> %rdx.minmax.select, <16 x i8> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i8> %rdx.minmax.select23, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp slt <16 x i8> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i8> %rdx.minmax.select23, <16 x i8> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i8> %rdx.minmax.select26, <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp slt <16 x i8> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i8> %rdx.minmax.select26, <16 x i8> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i8> %rdx.minmax.select29, <16 x i8> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp slt <16 x i8> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i8> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i8> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i8 %rdx.minmax.select29.elt, i8 %rdx.shuf30.elt
  ret i8 %r
}

; CHECK-LABEL: smin_H
; CHECK: sminv {{h[0-9]+}}, {{v[0-9]+}}.8h
define i16 @smin_H(<8 x i16>* nocapture readonly %arr) {
  %rdx.minmax.select = load <8 x i16>, <8 x i16>* %arr
  %rdx.shuf = shufflevector <8 x i16> %rdx.minmax.select, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp23 = icmp slt <8 x i16> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select24 = select <8 x i1> %rdx.minmax.cmp23, <8 x i16> %rdx.minmax.select, <8 x i16> %rdx.shuf
  %rdx.shuf25 = shufflevector <8 x i16> %rdx.minmax.select24, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp26 = icmp slt <8 x i16> %rdx.minmax.select24, %rdx.shuf25
  %rdx.minmax.select27 = select <8 x i1> %rdx.minmax.cmp26, <8 x i16> %rdx.minmax.select24, <8 x i16> %rdx.shuf25
  %rdx.shuf28 = shufflevector <8 x i16> %rdx.minmax.select27, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp29 = icmp slt <8 x i16> %rdx.minmax.select27, %rdx.shuf28
  %rdx.minmax.cmp29.elt = extractelement <8 x i1> %rdx.minmax.cmp29, i32 0
  %rdx.minmax.select27.elt = extractelement <8 x i16> %rdx.minmax.select27, i32 0
  %rdx.shuf28.elt = extractelement <8 x i16> %rdx.minmax.select27, i32 1
  %r = select i1 %rdx.minmax.cmp29.elt, i16 %rdx.minmax.select27.elt, i16 %rdx.shuf28.elt
  ret i16 %r
}

; CHECK-LABEL: smin_S
; CHECK: sminv {{s[0-9]+}}, {{v[0-9]+}}.4s
define i32 @smin_S(<4 x i32>* nocapture readonly %arr) {
  %rdx.minmax.select = load <4 x i32>, <4 x i32>* %arr
  %rdx.shuf = shufflevector <4 x i32> %rdx.minmax.select, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp18 = icmp slt <4 x i32> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select19 = select <4 x i1> %rdx.minmax.cmp18, <4 x i32> %rdx.minmax.select, <4 x i32> %rdx.shuf
  %rdx.shuf20 = shufflevector <4 x i32> %rdx.minmax.select19, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp21 = icmp slt <4 x i32> %rdx.minmax.select19, %rdx.shuf20
  %rdx.minmax.cmp21.elt = extractelement <4 x i1> %rdx.minmax.cmp21, i32 0
  %rdx.minmax.select19.elt = extractelement <4 x i32> %rdx.minmax.select19, i32 0
  %rdx.shuf20.elt = extractelement <4 x i32> %rdx.minmax.select19, i32 1
  %r = select i1 %rdx.minmax.cmp21.elt, i32 %rdx.minmax.select19.elt, i32 %rdx.shuf20.elt
  ret i32 %r
}

; CHECK-LABEL: smin_D
; CHECK-NOT: sminv
define i64 @smin_D(<2 x i64>* nocapture readonly %arr) {
  %rdx.minmax.select = load <2 x i64>, <2 x i64>* %arr
  %rdx.shuf = shufflevector <2 x i64> %rdx.minmax.select, <2 x i64> undef, <2 x i32> <i32 1, i32 undef>
  %rdx.minmax.cmp18 = icmp slt <2 x i64> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.cmp18.elt = extractelement <2 x i1> %rdx.minmax.cmp18, i32 0
  %rdx.minmax.select.elt = extractelement <2 x i64> %rdx.minmax.select, i32 0
  %rdx.shuf.elt = extractelement <2 x i64> %rdx.minmax.select, i32 1
  %r = select i1 %rdx.minmax.cmp18.elt, i64 %rdx.minmax.select.elt, i64 %rdx.shuf.elt
  ret i64 %r
}


; CHECK-LABEL: umin_B
; CHECK: uminv {{b[0-9]+}}, {{v[0-9]+}}.16b
define i8 @umin_B(<16 x i8>* nocapture readonly %arr)  {
  %rdx.minmax.select = load <16 x i8>, <16 x i8>* %arr
  %rdx.shuf = shufflevector <16 x i8> %rdx.minmax.select, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp ult <16 x i8> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i8> %rdx.minmax.select, <16 x i8> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i8> %rdx.minmax.select23, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp ult <16 x i8> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i8> %rdx.minmax.select23, <16 x i8> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i8> %rdx.minmax.select26, <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp ult <16 x i8> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i8> %rdx.minmax.select26, <16 x i8> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i8> %rdx.minmax.select29, <16 x i8> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp ult <16 x i8> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i8> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i8> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i8 %rdx.minmax.select29.elt, i8 %rdx.shuf30.elt
  ret i8 %r
}

; CHECK-LABEL: umin_H
; CHECK: uminv {{h[0-9]+}}, {{v[0-9]+}}.8h
define i16 @umin_H(<8 x i16>* nocapture readonly %arr)  {
  %rdx.minmax.select = load <8 x i16>, <8 x i16>* %arr
  %rdx.shuf = shufflevector <8 x i16> %rdx.minmax.select, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp23 = icmp ult <8 x i16> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select24 = select <8 x i1> %rdx.minmax.cmp23, <8 x i16> %rdx.minmax.select, <8 x i16> %rdx.shuf
  %rdx.shuf25 = shufflevector <8 x i16> %rdx.minmax.select24, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp26 = icmp ult <8 x i16> %rdx.minmax.select24, %rdx.shuf25
  %rdx.minmax.select27 = select <8 x i1> %rdx.minmax.cmp26, <8 x i16> %rdx.minmax.select24, <8 x i16> %rdx.shuf25
  %rdx.shuf28 = shufflevector <8 x i16> %rdx.minmax.select27, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp29 = icmp ult <8 x i16> %rdx.minmax.select27, %rdx.shuf28
  %rdx.minmax.cmp29.elt = extractelement <8 x i1> %rdx.minmax.cmp29, i32 0
  %rdx.minmax.select27.elt = extractelement <8 x i16> %rdx.minmax.select27, i32 0
  %rdx.shuf28.elt = extractelement <8 x i16> %rdx.minmax.select27, i32 1
  %r = select i1 %rdx.minmax.cmp29.elt, i16 %rdx.minmax.select27.elt, i16 %rdx.shuf28.elt
  ret i16 %r
}

; CHECK-LABEL: umin_S
; CHECK: uminv {{s[0-9]+}}, {{v[0-9]+}}.4s
define i32 @umin_S(<4 x i32>* nocapture readonly %arr) {
  %rdx.minmax.select = load <4 x i32>, <4 x i32>* %arr
  %rdx.shuf = shufflevector <4 x i32> %rdx.minmax.select, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp18 = icmp ult <4 x i32> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select19 = select <4 x i1> %rdx.minmax.cmp18, <4 x i32> %rdx.minmax.select, <4 x i32> %rdx.shuf
  %rdx.shuf20 = shufflevector <4 x i32> %rdx.minmax.select19, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp21 = icmp ult <4 x i32> %rdx.minmax.select19, %rdx.shuf20
  %rdx.minmax.cmp21.elt = extractelement <4 x i1> %rdx.minmax.cmp21, i32 0
  %rdx.minmax.select19.elt = extractelement <4 x i32> %rdx.minmax.select19, i32 0
  %rdx.shuf20.elt = extractelement <4 x i32> %rdx.minmax.select19, i32 1
  %r = select i1 %rdx.minmax.cmp21.elt, i32 %rdx.minmax.select19.elt, i32 %rdx.shuf20.elt
  ret i32 %r
}

; CHECK-LABEL: umin_D
; CHECK-NOT: uminv
define i64 @umin_D(<2 x i64>* nocapture readonly %arr)  {
  %rdx.minmax.select = load <2 x i64>, <2 x i64>* %arr
  %rdx.shuf = shufflevector <2 x i64> %rdx.minmax.select, <2 x i64> undef, <2 x i32> <i32 1, i32 undef>
  %rdx.minmax.cmp18 = icmp ult <2 x i64> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.cmp18.elt = extractelement <2 x i1> %rdx.minmax.cmp18, i32 0
  %rdx.minmax.select.elt = extractelement <2 x i64> %rdx.minmax.select, i32 0
  %rdx.shuf.elt = extractelement <2 x i64> %rdx.minmax.select, i32 1
  %r = select i1 %rdx.minmax.cmp18.elt, i64 %rdx.minmax.select.elt, i64 %rdx.shuf.elt
  ret i64 %r
}

; CHECK-LABEL: fmaxnm_S
; CHECK: fmaxnmv
define float @fmaxnm_S(<4 x float>* nocapture readonly %arr) {
  %rdx.minmax.select  = load <4 x float>, <4 x float>* %arr
  %rdx.shuf = shufflevector <4 x float> %rdx.minmax.select, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp = fcmp fast oge <4 x float> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select1 = select <4 x i1> %rdx.minmax.cmp, <4 x float> %rdx.minmax.select, <4 x float> %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x float> %rdx.minmax.select1, <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp1 = fcmp fast oge <4 x float> %rdx.minmax.select1, %rdx.shuf1
  %rdx.minmax.cmp1.elt = extractelement <4 x i1> %rdx.minmax.cmp1, i32 0
  %rdx.minmax.select1.elt = extractelement <4 x float> %rdx.minmax.select1, i32 0
  %rdx.shuf1.elt = extractelement <4 x float> %rdx.minmax.select1, i32 1
  %r = select i1 %rdx.minmax.cmp1.elt, float %rdx.minmax.select1.elt, float %rdx.shuf1.elt
  ret float %r
}

; CHECK-LABEL: fminnm_S
; CHECK: fminnmv
define float @fminnm_S(<4 x float>* nocapture readonly %arr) {
  %rdx.minmax.select  = load <4 x float>, <4 x float>* %arr
  %rdx.shuf = shufflevector <4 x float> %rdx.minmax.select, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp = fcmp fast ole <4 x float> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select1 = select <4 x i1> %rdx.minmax.cmp, <4 x float> %rdx.minmax.select, <4 x float> %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x float> %rdx.minmax.select1, <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp1 = fcmp fast ole <4 x float> %rdx.minmax.select1, %rdx.shuf1
  %rdx.minmax.cmp1.elt = extractelement <4 x i1> %rdx.minmax.cmp1, i32 0
  %rdx.minmax.select1.elt = extractelement <4 x float> %rdx.minmax.select1, i32 0
  %rdx.shuf1.elt = extractelement <4 x float> %rdx.minmax.select1, i32 1
  %r = select i1 %rdx.minmax.cmp1.elt, float %rdx.minmax.select1.elt, float %rdx.shuf1.elt
  ret float %r
}

define i16 @oversized_umax_256(<16 x i16>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_umax_256
; CHECK: umax [[V0:v[0-9]+]].8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: umaxv {{h[0-9]+}}, [[V0]]
  %rdx.minmax.select = load <16 x i16>, <16 x i16>* %arr
  %rdx.shuf = shufflevector <16 x i16> %rdx.minmax.select, <16 x i16> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp ugt <16 x i16> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i16> %rdx.minmax.select, <16 x i16> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i16> %rdx.minmax.select23, <16 x i16> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp ugt <16 x i16> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i16> %rdx.minmax.select23, <16 x i16> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i16> %rdx.minmax.select26, <16 x i16> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp ugt <16 x i16> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i16> %rdx.minmax.select26, <16 x i16> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i16> %rdx.minmax.select29, <16 x i16> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp ugt <16 x i16> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i16> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i16> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i16 %rdx.minmax.select29.elt, i16 %rdx.shuf30.elt
  ret i16 %r
}

define i32 @oversized_umax_512(<16 x i32>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_umax_512
; CHECK: umax v
; CHECK-NEXT: umax v
; CHECK-NEXT: umax [[V0:v[0-9]+]].4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-NEXT: umaxv {{s[0-9]+}}, [[V0]]
  %arr.load = load <16 x i32>, <16 x i32>* %arr
  %rdx.shuf = shufflevector <16 x i32> %arr.load, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp ugt <16 x i32> %arr.load, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i32> %arr.load, <16 x i32> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i32> %rdx.minmax.select23, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp ugt <16 x i32> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i32> %rdx.minmax.select23, <16 x i32> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i32> %rdx.minmax.select26, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp ugt <16 x i32> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i32> %rdx.minmax.select26, <16 x i32> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i32> %rdx.minmax.select29, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp ugt <16 x i32> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i32> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i32> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i32 %rdx.minmax.select29.elt, i32 %rdx.shuf30.elt
  ret i32 %r
}

define i16 @oversized_umin_256(<16 x i16>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_umin_256
; CHECK: umin [[V0:v[0-9]+]].8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: uminv {{h[0-9]+}}, [[V0]]
  %rdx.minmax.select = load <16 x i16>, <16 x i16>* %arr
  %rdx.shuf = shufflevector <16 x i16> %rdx.minmax.select, <16 x i16> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp ult <16 x i16> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i16> %rdx.minmax.select, <16 x i16> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i16> %rdx.minmax.select23, <16 x i16> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp ult <16 x i16> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i16> %rdx.minmax.select23, <16 x i16> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i16> %rdx.minmax.select26, <16 x i16> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp ult <16 x i16> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i16> %rdx.minmax.select26, <16 x i16> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i16> %rdx.minmax.select29, <16 x i16> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp ult <16 x i16> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i16> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i16> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i16 %rdx.minmax.select29.elt, i16 %rdx.shuf30.elt
  ret i16 %r
}

define i32 @oversized_umin_512(<16 x i32>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_umin_512
; CHECK: umin v
; CHECK-NEXT: umin v
; CHECK-NEXT: umin [[V0:v[0-9]+]].4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-NEXT: uminv {{s[0-9]+}}, [[V0]]
  %arr.load = load <16 x i32>, <16 x i32>* %arr
  %rdx.shuf = shufflevector <16 x i32> %arr.load, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp ult <16 x i32> %arr.load, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i32> %arr.load, <16 x i32> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i32> %rdx.minmax.select23, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp ult <16 x i32> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i32> %rdx.minmax.select23, <16 x i32> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i32> %rdx.minmax.select26, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp ult <16 x i32> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i32> %rdx.minmax.select26, <16 x i32> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i32> %rdx.minmax.select29, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp ult <16 x i32> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i32> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i32> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i32 %rdx.minmax.select29.elt, i32 %rdx.shuf30.elt
  ret i32 %r
}

define i16 @oversized_smax_256(<16 x i16>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_smax_256
; CHECK: smax [[V0:v[0-9]+]].8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: smaxv {{h[0-9]+}}, [[V0]]
  %arr.load = load <16 x i16>, <16 x i16>* %arr
  %rdx.shuf = shufflevector <16 x i16> %arr.load, <16 x i16> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp sgt <16 x i16> %arr.load, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i16> %arr.load, <16 x i16> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i16> %rdx.minmax.select23, <16 x i16> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp sgt <16 x i16> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i16> %rdx.minmax.select23, <16 x i16> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i16> %rdx.minmax.select26, <16 x i16> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp sgt <16 x i16> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i16> %rdx.minmax.select26, <16 x i16> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i16> %rdx.minmax.select29, <16 x i16> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp sgt <16 x i16> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i16> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i16> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i16 %rdx.minmax.select29.elt, i16 %rdx.shuf30.elt
  ret i16 %r
}

define i32 @oversized_smax_512(<16 x i32>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_smax_512
; CHECK: smax v
; CHECK-NEXT: smax v
; CHECK-NEXT: smax [[V0:v[0-9]+]].4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-NEXT: smaxv {{s[0-9]+}}, [[V0]]
  %arr.load = load <16 x i32>, <16 x i32>* %arr
  %rdx.shuf = shufflevector <16 x i32> %arr.load, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp sgt <16 x i32> %arr.load, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i32> %arr.load, <16 x i32> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i32> %rdx.minmax.select23, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp sgt <16 x i32> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i32> %rdx.minmax.select23, <16 x i32> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i32> %rdx.minmax.select26, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp sgt <16 x i32> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i32> %rdx.minmax.select26, <16 x i32> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i32> %rdx.minmax.select29, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp sgt <16 x i32> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i32> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i32> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i32 %rdx.minmax.select29.elt, i32 %rdx.shuf30.elt
  ret i32 %r
}

define i16 @oversized_smin_256(<16 x i16>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_smin_256
; CHECK: smin [[V0:v[0-9]+]].8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: sminv {{h[0-9]+}}, [[V0]]
  %rdx.minmax.select = load <16 x i16>, <16 x i16>* %arr
  %rdx.shuf = shufflevector <16 x i16> %rdx.minmax.select, <16 x i16> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp slt <16 x i16> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i16> %rdx.minmax.select, <16 x i16> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i16> %rdx.minmax.select23, <16 x i16> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp slt <16 x i16> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i16> %rdx.minmax.select23, <16 x i16> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i16> %rdx.minmax.select26, <16 x i16> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp slt <16 x i16> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i16> %rdx.minmax.select26, <16 x i16> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i16> %rdx.minmax.select29, <16 x i16> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp slt <16 x i16> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i16> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i16> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i16 %rdx.minmax.select29.elt, i16 %rdx.shuf30.elt
  ret i16 %r
}

define i32 @oversized_smin_512(<16 x i32>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_smin_512
; CHECK: smin v
; CHECK-NEXT: smin v
; CHECK-NEXT: smin [[V0:v[0-9]+]].4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-NEXT: sminv {{s[0-9]+}}, [[V0]]
  %arr.load = load <16 x i32>, <16 x i32>* %arr
  %rdx.shuf = shufflevector <16 x i32> %arr.load, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp22 = icmp slt <16 x i32> %arr.load, %rdx.shuf
  %rdx.minmax.select23 = select <16 x i1> %rdx.minmax.cmp22, <16 x i32> %arr.load, <16 x i32> %rdx.shuf
  %rdx.shuf24 = shufflevector <16 x i32> %rdx.minmax.select23, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp25 = icmp slt <16 x i32> %rdx.minmax.select23, %rdx.shuf24
  %rdx.minmax.select26 = select <16 x i1> %rdx.minmax.cmp25, <16 x i32> %rdx.minmax.select23, <16 x i32> %rdx.shuf24
  %rdx.shuf27 = shufflevector <16 x i32> %rdx.minmax.select26, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp28 = icmp slt <16 x i32> %rdx.minmax.select26, %rdx.shuf27
  %rdx.minmax.select29 = select <16 x i1> %rdx.minmax.cmp28, <16 x i32> %rdx.minmax.select26, <16 x i32> %rdx.shuf27
  %rdx.shuf30 = shufflevector <16 x i32> %rdx.minmax.select29, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp31 = icmp slt <16 x i32> %rdx.minmax.select29, %rdx.shuf30
  %rdx.minmax.cmp31.elt = extractelement <16 x i1> %rdx.minmax.cmp31, i32 0
  %rdx.minmax.select29.elt = extractelement <16 x i32> %rdx.minmax.select29, i32 0
  %rdx.shuf30.elt = extractelement <16 x i32> %rdx.minmax.select29, i32 1
  %r = select i1 %rdx.minmax.cmp31.elt, i32 %rdx.minmax.select29.elt, i32 %rdx.shuf30.elt
  ret i32 %r
}
