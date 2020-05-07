; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; SDIV
;

define <vscale x 4 x i32> @sdiv_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @sdiv_i32
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: sdiv z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %div = sdiv <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %div
}

define <vscale x 2 x i64> @sdiv_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: @sdiv_i64
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: sdiv z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %div = sdiv <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %div
}

define <vscale x 8 x i32> @sdiv_split_i32(<vscale x 8 x i32> %a, <vscale x 8 x i32> %b) {
; CHECK-LABEL: @sdiv_split_i32
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: sdiv z0.s, p0/m, z0.s, z2.s
; CHECK-DAG: sdiv z1.s, p0/m, z1.s, z3.s
; CHECK-NEXT: ret
  %div = sdiv <vscale x 8 x i32> %a, %b
  ret <vscale x 8 x i32> %div
}

define <vscale x 2 x i32> @sdiv_widen_i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b) {
; CHECK-LABEL: @sdiv_widen_i32
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: sxtw z1.d, p0/m, z1.d
; CHECK-DAG: sxtw z0.d, p0/m, z0.d
; CHECK-DAG: sdiv z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %div = sdiv <vscale x 2 x i32> %a, %b
  ret <vscale x 2 x i32> %div
}

define <vscale x 4 x i64> @sdiv_split_i64(<vscale x 4 x i64> %a, <vscale x 4 x i64> %b) {
; CHECK-LABEL: @sdiv_split_i64
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: sdiv z0.d, p0/m, z0.d, z2.d
; CHECK-DAG: sdiv z1.d, p0/m, z1.d, z3.d
; CHECK-NEXT: ret
  %div = sdiv <vscale x 4 x i64> %a, %b
  ret <vscale x 4 x i64> %div
}

;
; UDIV
;

define <vscale x 4 x i32> @udiv_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @udiv_i32
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: udiv z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %div = udiv <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %div
}

define <vscale x 2 x i64> @udiv_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: @udiv_i64
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: udiv z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %div = udiv <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %div
}

define <vscale x 8 x i32> @udiv_split_i32(<vscale x 8 x i32> %a, <vscale x 8 x i32> %b) {
; CHECK-LABEL: @udiv_split_i32
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: udiv z0.s, p0/m, z0.s, z2.s
; CHECK-DAG: udiv z1.s, p0/m, z1.s, z3.s
; CHECK-NEXT: ret
  %div = udiv <vscale x 8 x i32> %a, %b
  ret <vscale x 8 x i32> %div
}

define <vscale x 2 x i32> @udiv_widen_i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b) {
; CHECK-LABEL: @udiv_widen_i32
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: and z1.d, z1.d, #0xffffffff
; CHECK-DAG: and z0.d, z0.d, #0xffffffff
; CHECK-DAG: udiv z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %div = udiv <vscale x 2 x i32> %a, %b
  ret <vscale x 2 x i32> %div
}

define <vscale x 4 x i64> @udiv_split_i64(<vscale x 4 x i64> %a, <vscale x 4 x i64> %b) {
; CHECK-LABEL: @udiv_split_i64
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: udiv z0.d, p0/m, z0.d, z2.d
; CHECK-DAG: udiv z1.d, p0/m, z1.d, z3.d
; CHECK-NEXT: ret
  %div = udiv <vscale x 4 x i64> %a, %b
  ret <vscale x 4 x i64> %div
}

;
; SMIN
;

define <vscale x 16 x i8> @smin_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: @smin_i8
; CHECK-DAG: ptrue p0.b
; CHECK-DAG: smin z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %cmp = icmp slt <vscale x 16 x i8> %a, %b
  %min = select <vscale x 16 x i1> %cmp, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b
  ret <vscale x 16 x i8> %min
}

define <vscale x 8 x i16> @smin_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: @smin_i16
; CHECK-DAG: ptrue p0.h
; CHECK-DAG: smin z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %cmp = icmp slt <vscale x 8 x i16> %a, %b
  %min = select <vscale x 8 x i1> %cmp, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b
  ret <vscale x 8 x i16> %min
}

define <vscale x 4 x i32> @smin_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: smin_i32:
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: smin z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %cmp = icmp slt <vscale x 4 x i32> %a, %b
  %min = select <vscale x 4 x i1> %cmp, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b
  ret <vscale x 4 x i32> %min
}

define <vscale x 2 x i64> @smin_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: smin_i64:
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: smin z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %cmp = icmp slt <vscale x 2 x i64> %a, %b
  %min = select <vscale x 2 x i1> %cmp, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b
  ret <vscale x 2 x i64> %min
}

;
; UMIN
;

define <vscale x 16 x i8> @umin_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: @umin_i8
; CHECK-DAG: ptrue p0.b
; CHECK-DAG: umin z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %cmp = icmp ult <vscale x 16 x i8> %a, %b
  %min = select <vscale x 16 x i1> %cmp, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b
  ret <vscale x 16 x i8> %min
}

define <vscale x 8 x i16> @umin_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: @umin_i16
; CHECK-DAG: ptrue p0.h
; CHECK-DAG: umin z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %cmp = icmp ult <vscale x 8 x i16> %a, %b
  %min = select <vscale x 8 x i1> %cmp, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b
  ret <vscale x 8 x i16> %min
}

define <vscale x 4 x i32> @umin_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: umin_i32:
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: umin z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %cmp = icmp ult <vscale x 4 x i32> %a, %b
  %min = select <vscale x 4 x i1> %cmp, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b
  ret <vscale x 4 x i32> %min
}

define <vscale x 2 x i64> @umin_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: umin_i64:
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: umin z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %cmp = icmp ult <vscale x 2 x i64> %a, %b
  %min = select <vscale x 2 x i1> %cmp, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b
  ret <vscale x 2 x i64> %min
}

;
; SMAX
;

define <vscale x 16 x i8> @smax_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: @smax_i8
; CHECK-DAG: ptrue p0.b
; CHECK-DAG: smax z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %cmp = icmp sgt <vscale x 16 x i8> %a, %b
  %min = select <vscale x 16 x i1> %cmp, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b
  ret <vscale x 16 x i8> %min
}

define <vscale x 8 x i16> @smax_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: @smax_i16
; CHECK-DAG: ptrue p0.h
; CHECK-DAG: smax z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %cmp = icmp sgt <vscale x 8 x i16> %a, %b
  %min = select <vscale x 8 x i1> %cmp, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b
  ret <vscale x 8 x i16> %min
}

define <vscale x 4 x i32> @smax_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: smax_i32:
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: smax z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %cmp = icmp sgt <vscale x 4 x i32> %a, %b
  %min = select <vscale x 4 x i1> %cmp, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b
  ret <vscale x 4 x i32> %min
}

define <vscale x 2 x i64> @smax_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: smax_i64:
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: smax z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %cmp = icmp sgt <vscale x 2 x i64> %a, %b
  %min = select <vscale x 2 x i1> %cmp, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b
  ret <vscale x 2 x i64> %min
}

;
; UMAX
;

define <vscale x 16 x i8> @umax_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: @umax_i8
; CHECK-DAG: ptrue p0.b
; CHECK-DAG: umax z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %cmp = icmp ugt <vscale x 16 x i8> %a, %b
  %min = select <vscale x 16 x i1> %cmp, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b
  ret <vscale x 16 x i8> %min
}

define <vscale x 8 x i16> @umax_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: @umax_i16
; CHECK-DAG: ptrue p0.h
; CHECK-DAG: umax z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %cmp = icmp ugt <vscale x 8 x i16> %a, %b
  %min = select <vscale x 8 x i1> %cmp, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b
  ret <vscale x 8 x i16> %min
}

define <vscale x 4 x i32> @umax_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: umax_i32:
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: umax z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %cmp = icmp ugt <vscale x 4 x i32> %a, %b
  %min = select <vscale x 4 x i1> %cmp, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b
  ret <vscale x 4 x i32> %min
}

define <vscale x 2 x i64> @umax_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: umax_i64:
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: umax z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %cmp = icmp ugt <vscale x 2 x i64> %a, %b
  %min = select <vscale x 2 x i1> %cmp, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b
  ret <vscale x 2 x i64> %min
}

;
; ASR
;

define <vscale x 16 x i8> @asr_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b){
; CHECK-LABEL: @asr_i8
; CHECK-DAG: ptrue p0.b
; CHECK-DAG: asr z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %shr = ashr <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i8> %shr
}

define <vscale x 8 x i16> @asr_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b){
; CHECK-LABEL: @asr_i16
; CHECK-DAG: ptrue p0.h
; CHECK-DAG: asr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %shr = ashr <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i16> %shr
}

define <vscale x 4 x i32> @asr_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b){
; CHECK-LABEL: @asr_i32
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: asr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %shr = ashr <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %shr
}

define <vscale x 2 x i64> @asr_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b){
; CHECK-LABEL: @asr_i64
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: asr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %shr = ashr <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %shr
}

define <vscale x 16 x i16> @asr_split_i16(<vscale x 16 x i16> %a, <vscale x 16 x i16> %b){
; CHECK-LABEL: @asr_split_i16
; CHECK-DAG: ptrue p0.h
; CHECK-DAG: asr z0.h, p0/m, z0.h, z2.h
; CHECK-DAG: asr z1.h, p0/m, z1.h, z3.h
; CHECK-NEXT: ret
  %shr = ashr <vscale x 16 x i16> %a, %b
  ret <vscale x 16 x i16> %shr
}

define <vscale x 2 x i32> @asr_promote_i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b){
; CHECK-LABEL: @asr_promote_i32
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: and z1.d, z1.d, #0xffffffff
; CHECK-DAG: asr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %shr = ashr <vscale x 2 x i32> %a, %b
  ret <vscale x 2 x i32> %shr
}

;
; LSL
;

define <vscale x 16 x i8> @lsl_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b){
; CHECK-LABEL: @lsl_i8
; CHECK-DAG: ptrue p0.b
; CHECK-DAG: lsl z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %shl = shl <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i8> %shl
}

define <vscale x 8 x i16> @lsl_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b){
; CHECK-LABEL: @lsl_i16
; CHECK-DAG: ptrue p0.h
; CHECK-DAG: lsl z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %shl = shl <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i16> %shl
}

define <vscale x 4 x i32> @lsl_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b){
; CHECK-LABEL: @lsl_i32
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: lsl z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %shl = shl <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %shl
}

define <vscale x 2 x i64> @lsl_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b){
; CHECK-LABEL: @lsl_i64
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: lsl z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %shl = shl <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %shl
}

define <vscale x 4 x i64> @lsl_split_i64(<vscale x 4 x i64> %a, <vscale x 4 x i64> %b){
; CHECK-LABEL: @lsl_split_i64
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: lsl z0.d, p0/m, z0.d, z2.d
; CHECK-DAG: lsl z1.d, p0/m, z1.d, z3.d
; CHECK-NEXT: ret
  %shl = shl <vscale x 4 x i64> %a, %b
  ret <vscale x 4 x i64> %shl
}

define <vscale x 4 x i16> @lsl_promote_i16(<vscale x 4 x i16> %a, <vscale x 4 x i16> %b){
; CHECK-LABEL: @lsl_promote_i16
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: and z1.s, z1.s, #0xffff
; CHECK-DAG: lsl z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %shl = shl <vscale x 4 x i16> %a, %b
  ret <vscale x 4 x i16> %shl
}

;
; LSR
;

define <vscale x 16 x i8> @lsr_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b){
; CHECK-LABEL: @lsr_i8
; CHECK-DAG: ptrue p0.b
; CHECK-DAG: lsr z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %shr = lshr <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i8> %shr
}

define <vscale x 8 x i16> @lsr_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b){
; CHECK-LABEL: @lsr_i16
; CHECK-DAG: ptrue p0.h
; CHECK-DAG: lsr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %shr = lshr <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i16> %shr
}

define <vscale x 4 x i32> @lsr_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b){
; CHECK-LABEL: @lsr_i32
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: lsr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %shr = lshr <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %shr
}

define <vscale x 2 x i64> @lsr_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b){
; CHECK-LABEL: @lsr_i64
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: lsr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %shr = lshr <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %shr
}

define <vscale x 8 x i8> @lsr_promote_i8(<vscale x 8 x i8> %a, <vscale x 8 x i8> %b){
; CHECK-LABEL: @lsr_promote_i8
; CHECK-DAG: ptrue p0.h
; CHECK-DAG: and z1.h, z1.h, #0xff
; CHECK-DAG: lsr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %shr = lshr <vscale x 8 x i8> %a, %b
  ret <vscale x 8 x i8> %shr
}

define <vscale x 8 x i32> @lsr_split_i32(<vscale x 8 x i32> %a, <vscale x 8 x i32> %b){
; CHECK-LABEL: @lsr_split_i32
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: lsr z0.s, p0/m, z0.s, z2.s
; CHECK-DAG: lsr z1.s, p0/m, z1.s, z3.s
; CHECK-NEXT: ret
  %shr = lshr <vscale x 8 x i32> %a, %b
  ret <vscale x 8 x i32> %shr
}
