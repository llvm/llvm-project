; RUN: llc < %s -verify-machineinstrs -spiller=trivial
; RUN: llc < %s -verify-machineinstrs -spiller=inline
; PR8612
;
; This test has an inline asm with early-clobber arguments.
; It is big enough that one of the early clobber registers is spilled.
;
; All the spillers would get the live ranges wrong when spilling an early
; clobber, allowing the undef register to be allocated to the same register as
; the early clobber.
;
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32"
target triple = "armv7-eabi"

%0 = type { i32, i32 }

define void @foo(i32* %in) nounwind {
entry:
  br label %bb.i

bb.i:                                             ; preds = %bb.i, %entry
  br i1 undef, label %bb10.preheader.i, label %bb.i

bb10.preheader.i:                                 ; preds = %bb.i
  br label %bb10.i

bb10.i:                                           ; preds = %bb10.i, %bb10.preheader.i
  br i1 undef, label %bb27.i, label %bb10.i

bb27.i:                                           ; preds = %bb10.i
  br label %bb28.i

bb28.i:                                           ; preds = %bb28.i, %bb27.i
  br i1 undef, label %presymmetry.exit, label %bb28.i

presymmetry.exit:                                 ; preds = %bb28.i
  %tmp175387 = or i32 undef, 12
  %scevgep101.i = getelementptr i32* %in, i32 undef
  %tmp189401 = or i32 undef, 7
  %scevgep97.i = getelementptr i32* %in, i32 undef
  %tmp198410 = or i32 undef, 1
  %scevgep.i48 = getelementptr i32* %in, i32 undef
  %0 = load i32* %scevgep.i48, align 4
  %1 = add nsw i32 %0, 0
  store i32 %1, i32* undef, align 4
  %asmtmp.i.i33.i.i.i = tail call %0 asm "smull\09$0, $1, $2, $3", "=&r,=&r,%r,r,~{cc}"(i32 undef, i32 1518500250) nounwind
  %asmresult1.i.i34.i.i.i = extractvalue %0 %asmtmp.i.i33.i.i.i, 1
  %2 = shl i32 %asmresult1.i.i34.i.i.i, 1
  %3 = load i32* null, align 4
  %4 = load i32* undef, align 4
  %5 = sub nsw i32 %3, %4
  %6 = load i32* undef, align 4
  %7 = load i32* null, align 4
  %8 = sub nsw i32 %6, %7
  %9 = load i32* %scevgep97.i, align 4
  %10 = load i32* undef, align 4
  %11 = sub nsw i32 %9, %10
  %12 = load i32* null, align 4
  %13 = load i32* %scevgep101.i, align 4
  %14 = sub nsw i32 %12, %13
  %15 = load i32* %scevgep.i48, align 4
  %16 = load i32* null, align 4
  %17 = add nsw i32 %16, %15
  %18 = sub nsw i32 %15, %16
  %19 = load i32* undef, align 4
  %20 = add nsw i32 %19, %2
  %21 = sub nsw i32 %19, %2
  %22 = add nsw i32 %14, %5
  %23 = sub nsw i32 %5, %14
  %24 = add nsw i32 %11, %8
  %25 = sub nsw i32 %8, %11
  %26 = add nsw i32 %21, %23
  store i32 %26, i32* %scevgep.i48, align 4
  %27 = sub nsw i32 %25, %18
  store i32 %27, i32* null, align 4
  %28 = sub nsw i32 %23, %21
  store i32 %28, i32* undef, align 4
  %29 = add nsw i32 %18, %25
  store i32 %29, i32* undef, align 4
  %30 = add nsw i32 %17, %22
  store i32 %30, i32* %scevgep101.i, align 4
  %31 = add nsw i32 %20, %24
  store i32 %31, i32* null, align 4
  unreachable
}
