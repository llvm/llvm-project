; RUN: opt -S %s -atomic-expand | FileCheck %s

;; Verify the cmpxchg and atomicrmw expansions where sub-word-size
;; instructions are not available.

;;; NOTE: this test is mostly target-independent -- any target which
;;; doesn't support cmpxchg of sub-word sizes would do.
target datalayout = "E-m:e-i64:64-n32:64-S128"
target triple = "sparcv9-unknown-unknown"

; CHECK-LABEL: @test_cmpxchg_i8(
; CHECK:  fence seq_cst
; CHECK:  %0 = ptrtoint i8* %arg to i64
; CHECK:  %1 = and i64 %0, -4
; CHECK:  %AlignedAddr = inttoptr i64 %1 to i32*
; CHECK:  %PtrLSB = and i64 %0, 3
; CHECK:  %2 = xor i64 %PtrLSB, 3
; CHECK:  %3 = shl i64 %2, 3
; CHECK:  %ShiftAmt = trunc i64 %3 to i32
; CHECK:  %Mask = shl i32 255, %ShiftAmt
; CHECK:  %Inv_Mask = xor i32 %Mask, -1
; CHECK:  %4 = zext i8 %new to i32
; CHECK:  %5 = shl i32 %4, %ShiftAmt
; CHECK:  %6 = zext i8 %old to i32
; CHECK:  %7 = shl i32 %6, %ShiftAmt
; CHECK:  %8 = load i32, i32* %AlignedAddr
; CHECK:  %9 = and i32 %8, %Inv_Mask
; CHECK:  br label %partword.cmpxchg.loop
; CHECK:partword.cmpxchg.loop:
; CHECK:  %10 = phi i32 [ %9, %entry ], [ %16, %partword.cmpxchg.failure ]
; CHECK:  %11 = or i32 %10, %5
; CHECK:  %12 = or i32 %10, %7
; CHECK:  %13 = cmpxchg i32* %AlignedAddr, i32 %12, i32 %11 monotonic monotonic
; CHECK:  %14 = extractvalue { i32, i1 } %13, 0
; CHECK:  %15 = extractvalue { i32, i1 } %13, 1
; CHECK:  br i1 %15, label %partword.cmpxchg.end, label %partword.cmpxchg.failure
; CHECK:partword.cmpxchg.failure:
; CHECK:  %16 = and i32 %14, %Inv_Mask
; CHECK:  %17 = icmp ne i32 %10, %16
; CHECK:  br i1 %17, label %partword.cmpxchg.loop, label %partword.cmpxchg.end
; CHECK:partword.cmpxchg.end:
; CHECK:  %18 = lshr i32 %14, %ShiftAmt
; CHECK:  %19 = trunc i32 %18 to i8
; CHECK:  %20 = insertvalue { i8, i1 } undef, i8 %19, 0
; CHECK:  %21 = insertvalue { i8, i1 } %20, i1 %15, 1
; CHECK:  fence seq_cst
; CHECK:  %ret = extractvalue { i8, i1 } %21, 0
; CHECK:  ret i8 %ret
define i8 @test_cmpxchg_i8(i8* %arg, i8 %old, i8 %new) {
entry:
  %ret_succ = cmpxchg i8* %arg, i8 %old, i8 %new seq_cst monotonic
  %ret = extractvalue { i8, i1 } %ret_succ, 0
  ret i8 %ret
}

; CHECK-LABEL: @test_cmpxchg_i16(
; CHECK:  fence seq_cst
; CHECK:  %0 = ptrtoint i16* %arg to i64
; CHECK:  %1 = and i64 %0, -4
; CHECK:  %AlignedAddr = inttoptr i64 %1 to i32*
; CHECK:  %PtrLSB = and i64 %0, 3
; CHECK:  %2 = xor i64 %PtrLSB, 2
; CHECK:  %3 = shl i64 %2, 3
; CHECK:  %ShiftAmt = trunc i64 %3 to i32
; CHECK:  %Mask = shl i32 65535, %ShiftAmt
; CHECK:  %Inv_Mask = xor i32 %Mask, -1
; CHECK:  %4 = zext i16 %new to i32
; CHECK:  %5 = shl i32 %4, %ShiftAmt
; CHECK:  %6 = zext i16 %old to i32
; CHECK:  %7 = shl i32 %6, %ShiftAmt
; CHECK:  %8 = load i32, i32* %AlignedAddr
; CHECK:  %9 = and i32 %8, %Inv_Mask
; CHECK:  br label %partword.cmpxchg.loop
; CHECK:partword.cmpxchg.loop:
; CHECK:  %10 = phi i32 [ %9, %entry ], [ %16, %partword.cmpxchg.failure ]
; CHECK:  %11 = or i32 %10, %5
; CHECK:  %12 = or i32 %10, %7
; CHECK:  %13 = cmpxchg i32* %AlignedAddr, i32 %12, i32 %11 monotonic monotonic
; CHECK:  %14 = extractvalue { i32, i1 } %13, 0
; CHECK:  %15 = extractvalue { i32, i1 } %13, 1
; CHECK:  br i1 %15, label %partword.cmpxchg.end, label %partword.cmpxchg.failure
; CHECK:partword.cmpxchg.failure:
; CHECK:  %16 = and i32 %14, %Inv_Mask
; CHECK:  %17 = icmp ne i32 %10, %16
; CHECK:  br i1 %17, label %partword.cmpxchg.loop, label %partword.cmpxchg.end
; CHECK:partword.cmpxchg.end:
; CHECK:  %18 = lshr i32 %14, %ShiftAmt
; CHECK:  %19 = trunc i32 %18 to i16
; CHECK:  %20 = insertvalue { i16, i1 } undef, i16 %19, 0
; CHECK:  %21 = insertvalue { i16, i1 } %20, i1 %15, 1
; CHECK:  fence seq_cst
; CHECK:  %ret = extractvalue { i16, i1 } %21, 0
; CHECK:  ret i16 %ret
define i16 @test_cmpxchg_i16(i16* %arg, i16 %old, i16 %new) {
entry:
  %ret_succ = cmpxchg i16* %arg, i16 %old, i16 %new seq_cst monotonic
  %ret = extractvalue { i16, i1 } %ret_succ, 0
  ret i16 %ret
}


; CHECK-LABEL: @test_add_i16(
; CHECK:  fence seq_cst
; CHECK:  %0 = ptrtoint i16* %arg to i64
; CHECK:  %1 = and i64 %0, -4
; CHECK:  %AlignedAddr = inttoptr i64 %1 to i32*
; CHECK:  %PtrLSB = and i64 %0, 3
; CHECK:  %2 = xor i64 %PtrLSB, 2
; CHECK:  %3 = shl i64 %2, 3
; CHECK:  %ShiftAmt = trunc i64 %3 to i32
; CHECK:  %Mask = shl i32 65535, %ShiftAmt
; CHECK:  %Inv_Mask = xor i32 %Mask, -1
; CHECK:  %4 = zext i16 %val to i32
; CHECK:  %ValOperand_Shifted = shl i32 %4, %ShiftAmt
; CHECK:  %5 = load i32, i32* %AlignedAddr, align 4
; CHECK:  br label %atomicrmw.start
; CHECK:atomicrmw.start:
; CHECK:  %loaded = phi i32 [ %5, %entry ], [ %newloaded, %atomicrmw.start ]
; CHECK:  %new = add i32 %loaded, %ValOperand_Shifted
; CHECK:  %6 = and i32 %new, %Mask
; CHECK:  %7 = and i32 %loaded, %Inv_Mask
; CHECK:  %8 = or i32 %7, %6
; CHECK:  %9 = cmpxchg i32* %AlignedAddr, i32 %loaded, i32 %8 monotonic monotonic
; CHECK:  %success = extractvalue { i32, i1 } %9, 1
; CHECK:  %newloaded = extractvalue { i32, i1 } %9, 0
; CHECK:  br i1 %success, label %atomicrmw.end, label %atomicrmw.start
; CHECK:atomicrmw.end:
; CHECK:  %10 = lshr i32 %newloaded, %ShiftAmt
; CHECK:  %11 = trunc i32 %10 to i16
; CHECK:  fence seq_cst
; CHECK:  ret i16 %11
define i16 @test_add_i16(i16* %arg, i16 %val) {
entry:
  %ret = atomicrmw add i16* %arg, i16 %val seq_cst
  ret i16 %ret
}

; CHECK-LABEL: @test_xor_i16(
; (I'm going to just assert on the bits that differ from add, above.)
; CHECK:atomicrmw.start:
; CHECK:  %new = xor i32 %loaded, %ValOperand_Shifted
; CHECK:  %6 = cmpxchg i32* %AlignedAddr, i32 %loaded, i32 %new monotonic monotonic
; CHECK:atomicrmw.end:
define i16 @test_xor_i16(i16* %arg, i16 %val) {
entry:
  %ret = atomicrmw xor i16* %arg, i16 %val seq_cst
  ret i16 %ret
}

; CHECK-LABEL: @test_min_i16(
; CHECK:atomicrmw.start:
; CHECK:  %6 = lshr i32 %loaded, %ShiftAmt
; CHECK:  %7 = trunc i32 %6 to i16
; CHECK:  %8 = icmp sle i16 %7, %val
; CHECK:  %new = select i1 %8, i16 %7, i16 %val
; CHECK:  %9 = zext i16 %new to i32
; CHECK:  %10 = shl i32 %9, %ShiftAmt
; CHECK:  %11 = and i32 %loaded, %Inv_Mask
; CHECK:  %12 = or i32 %11, %10
; CHECK:  %13 = cmpxchg i32* %AlignedAddr, i32 %loaded, i32 %12 monotonic monotonic
; CHECK:atomicrmw.end:
define i16 @test_min_i16(i16* %arg, i16 %val) {
entry:
  %ret = atomicrmw min i16* %arg, i16 %val seq_cst
  ret i16 %ret
}
