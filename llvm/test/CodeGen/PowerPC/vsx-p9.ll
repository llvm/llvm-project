; RUN: llc -verify-machineinstrs -mcpu=pwr9 \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 \
; RUN:   -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s

@uca = global <16 x i8> zeroinitializer, align 16
@ucb = global <16 x i8> zeroinitializer, align 16
@sca = global <16 x i8> zeroinitializer, align 16
@scb = global <16 x i8> zeroinitializer, align 16
@usa = global <8 x i16> zeroinitializer, align 16
@usb = global <8 x i16> zeroinitializer, align 16
@ssa = global <8 x i16> zeroinitializer, align 16
@ssb = global <8 x i16> zeroinitializer, align 16
@uia = global <4 x i32> zeroinitializer, align 16
@uib = global <4 x i32> zeroinitializer, align 16
@sia = global <4 x i32> zeroinitializer, align 16
@sib = global <4 x i32> zeroinitializer, align 16
@ulla = global <2 x i64> zeroinitializer, align 16
@ullb = global <2 x i64> zeroinitializer, align 16
@slla = global <2 x i64> zeroinitializer, align 16
@sllb = global <2 x i64> zeroinitializer, align 16
@uxa = global <1 x i128> zeroinitializer, align 16
@uxb = global <1 x i128> zeroinitializer, align 16
@sxa = global <1 x i128> zeroinitializer, align 16
@sxb = global <1 x i128> zeroinitializer, align 16
@vfa = global <4 x float> zeroinitializer, align 16
@vfb = global <4 x float> zeroinitializer, align 16
@vda = global <2 x double> zeroinitializer, align 16
@vdb = global <2 x double> zeroinitializer, align 16

define void @_Z4testv() {
entry:
; CHECK-LABEL: @_Z4testv
  %0 = load <16 x i8>, <16 x i8>* @uca, align 16
  %1 = load <16 x i8>, <16 x i8>* @ucb, align 16
  %add.i = add <16 x i8> %1, %0
  tail call void (...) @sink(<16 x i8> %add.i)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 4
; CHECK: vaddubm 2, 3, 2
; CHECK: stxvx 34,
; CHECK: bl sink
  %2 = load <16 x i8>, <16 x i8>* @sca, align 16
  %3 = load <16 x i8>, <16 x i8>* @scb, align 16
  %add.i22 = add <16 x i8> %3, %2
  tail call void (...) @sink(<16 x i8> %add.i22)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 4
; CHECK: vaddubm 2, 3, 2
; CHECK: stxvx 34,
; CHECK: bl sink
  %4 = load <8 x i16>, <8 x i16>* @usa, align 16
  %5 = load <8 x i16>, <8 x i16>* @usb, align 16
  %add.i21 = add <8 x i16> %5, %4
  tail call void (...) @sink(<8 x i16> %add.i21)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 4
; CHECK: vadduhm 2, 3, 2
; CHECK: stxvx 34,
; CHECK: bl sink
  %6 = load <8 x i16>, <8 x i16>* @ssa, align 16
  %7 = load <8 x i16>, <8 x i16>* @ssb, align 16
  %add.i20 = add <8 x i16> %7, %6
  tail call void (...) @sink(<8 x i16> %add.i20)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 4
; CHECK: vadduhm 2, 3, 2
; CHECK: stxvx 34,
; CHECK: bl sink
  %8 = load <4 x i32>, <4 x i32>* @uia, align 16
  %9 = load <4 x i32>, <4 x i32>* @uib, align 16
  %add.i19 = add <4 x i32> %9, %8
  tail call void (...) @sink(<4 x i32> %add.i19)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 4
; CHECK: vadduwm 2, 3, 2
; CHECK: stxvx 34,
; CHECK: bl sink
  %10 = load <4 x i32>, <4 x i32>* @sia, align 16
  %11 = load <4 x i32>, <4 x i32>* @sib, align 16
  %add.i18 = add <4 x i32> %11, %10
  tail call void (...) @sink(<4 x i32> %add.i18)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 4
; CHECK: vadduwm 2, 3, 2
; CHECK: stxvx 34,
; CHECK: bl sink
  %12 = load <2 x i64>, <2 x i64>* @ulla, align 16
  %13 = load <2 x i64>, <2 x i64>* @ullb, align 16
  %add.i17 = add <2 x i64> %13, %12
  tail call void (...) @sink(<2 x i64> %add.i17)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 4
; CHECK: vaddudm 2, 3, 2
; CHECK: stxvx 34,
; CHECK: bl sink
  %14 = load <2 x i64>, <2 x i64>* @slla, align 16
  %15 = load <2 x i64>, <2 x i64>* @sllb, align 16
  %add.i16 = add <2 x i64> %15, %14
  tail call void (...) @sink(<2 x i64> %add.i16)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 4
; CHECK: vaddudm 2, 3, 2
; CHECK: stxvx 34,
; CHECK: bl sink
  %16 = load <1 x i128>, <1 x i128>* @uxa, align 16
  %17 = load <1 x i128>, <1 x i128>* @uxb, align 16
  %add.i15 = add <1 x i128> %17, %16
  tail call void (...) @sink(<1 x i128> %add.i15)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 4
; CHECK: vadduqm 2, 3, 2
; CHECK: stxvx 34,
; CHECK: bl sink
  %18 = load <1 x i128>, <1 x i128>* @sxa, align 16
  %19 = load <1 x i128>, <1 x i128>* @sxb, align 16
  %add.i14 = add <1 x i128> %19, %18
  tail call void (...) @sink(<1 x i128> %add.i14)
; CHECK: lxvx 34, 0, 3
; CHECK: lxvx 35, 0, 4
; CHECK: vadduqm 2, 3, 2
; CHECK: stxvx 34,
; CHECK: bl sink
  %20 = load <4 x float>, <4 x float>* @vfa, align 16
  %21 = load <4 x float>, <4 x float>* @vfb, align 16
  %add.i13 = fadd <4 x float> %20, %21
  tail call void (...) @sink(<4 x float> %add.i13)
; CHECK: lxvx 0, 0, 3
; CHECK: lxvx 1, 0, 4
; CHECK: xvaddsp 34, 0, 1
; CHECK: stxvx 34,
; CHECK: bl sink
  %22 = load <2 x double>, <2 x double>* @vda, align 16
  %23 = load <2 x double>, <2 x double>* @vdb, align 16
  %add.i12 = fadd <2 x double> %22, %23
  tail call void (...) @sink(<2 x double> %add.i12)
; CHECK: lxvx 0, 0, 3
; CHECK: lxvx 1, 0, 4
; CHECK: xvadddp 0, 0, 1
; CHECK: stxvx 0,
; CHECK: bl sink
  ret void
}

declare void @sink(...)
