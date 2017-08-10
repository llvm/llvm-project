; RUN: llc < %s -verify-machineinstrs
;
; This test is disabled until PPCISelLowering learns to insert proper 64-bit
; code for ATOMIC_CMP_SWAP. Currently, it is inserting 32-bit instructions with
; 64-bit operands which causes the machine code verifier to throw a tantrum.
;
; XFAIL: *

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc64-apple-darwin9"

@sc = common global i8 0
@uc = common global i8 0
@ss = common global i16 0
@us = common global i16 0
@si = common global i32 0
@ui = common global i32 0
@sl = common global i64 0, align 8
@ul = common global i64 0, align 8
@sll = common global i64 0, align 8
@ull = common global i64 0, align 8

define void @test_op_ignore() nounwind {
entry:
  %0 = atomicrmw add i8* @sc, i8 1 monotonic
  %1 = atomicrmw add i8* @uc, i8 1 monotonic
  %2 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %3 = atomicrmw add i16* %2, i16 1 monotonic
  %4 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %5 = atomicrmw add i16* %4, i16 1 monotonic
  %6 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %7 = atomicrmw add i32* %6, i32 1 monotonic
  %8 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %9 = atomicrmw add i32* %8, i32 1 monotonic
  %10 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %11 = atomicrmw add i64* %10, i64 1 monotonic
  %12 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %13 = atomicrmw add i64* %12, i64 1 monotonic
  %14 = atomicrmw sub i8* @sc, i8 1 monotonic
  %15 = atomicrmw sub i8* @uc, i8 1 monotonic
  %16 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %17 = atomicrmw sub i16* %16, i16 1 monotonic
  %18 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %19 = atomicrmw sub i16* %18, i16 1 monotonic
  %20 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %21 = atomicrmw sub i32* %20, i32 1 monotonic
  %22 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %23 = atomicrmw sub i32* %22, i32 1 monotonic
  %24 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %25 = atomicrmw sub i64* %24, i64 1 monotonic
  %26 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %27 = atomicrmw sub i64* %26, i64 1 monotonic
  %28 = atomicrmw or i8* @sc, i8 1 monotonic
  %29 = atomicrmw or i8* @uc, i8 1 monotonic
  %30 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %31 = atomicrmw or i16* %30, i16 1 monotonic
  %32 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %33 = atomicrmw or i16* %32, i16 1 monotonic
  %34 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %35 = atomicrmw or i32* %34, i32 1 monotonic
  %36 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %37 = atomicrmw or i32* %36, i32 1 monotonic
  %38 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %39 = atomicrmw or i64* %38, i64 1 monotonic
  %40 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %41 = atomicrmw or i64* %40, i64 1 monotonic
  %42 = atomicrmw xor i8* @sc, i8 1 monotonic
  %43 = atomicrmw xor i8* @uc, i8 1 monotonic
  %44 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %45 = atomicrmw xor i16* %44, i16 1 monotonic
  %46 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %47 = atomicrmw xor i16* %46, i16 1 monotonic
  %48 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %49 = atomicrmw xor i32* %48, i32 1 monotonic
  %50 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %51 = atomicrmw xor i32* %50, i32 1 monotonic
  %52 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %53 = atomicrmw xor i64* %52, i64 1 monotonic
  %54 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %55 = atomicrmw xor i64* %54, i64 1 monotonic
  %56 = atomicrmw and i8* @sc, i8 1 monotonic
  %57 = atomicrmw and i8* @uc, i8 1 monotonic
  %58 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %59 = atomicrmw and i16* %58, i16 1 monotonic
  %60 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %61 = atomicrmw and i16* %60, i16 1 monotonic
  %62 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %63 = atomicrmw and i32* %62, i32 1 monotonic
  %64 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %65 = atomicrmw and i32* %64, i32 1 monotonic
  %66 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %67 = atomicrmw and i64* %66, i64 1 monotonic
  %68 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %69 = atomicrmw and i64* %68, i64 1 monotonic
  %70 = atomicrmw nand i8* @sc, i8 1 monotonic
  %71 = atomicrmw nand i8* @uc, i8 1 monotonic
  %72 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %73 = atomicrmw nand i16* %72, i16 1 monotonic
  %74 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %75 = atomicrmw nand i16* %74, i16 1 monotonic
  %76 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %77 = atomicrmw nand i32* %76, i32 1 monotonic
  %78 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %79 = atomicrmw nand i32* %78, i32 1 monotonic
  %80 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %81 = atomicrmw nand i64* %80, i64 1 monotonic
  %82 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %83 = atomicrmw nand i64* %82, i64 1 monotonic
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_fetch_and_op() nounwind {
entry:
  %0 = atomicrmw add i8* @sc, i8 11 monotonic
  store i8 %0, i8* @sc, align 1
  %1 = atomicrmw add i8* @uc, i8 11 monotonic
  store i8 %1, i8* @uc, align 1
  %2 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %3 = atomicrmw add i16* %2, i16 11 monotonic
  store i16 %3, i16* @ss, align 2
  %4 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %5 = atomicrmw add i16* %4, i16 11 monotonic
  store i16 %5, i16* @us, align 2
  %6 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %7 = atomicrmw add i32* %6, i32 11 monotonic
  store i32 %7, i32* @si, align 4
  %8 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %9 = atomicrmw add i32* %8, i32 11 monotonic
  store i32 %9, i32* @ui, align 4
  %10 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %11 = atomicrmw add i64* %10, i64 11 monotonic
  store i64 %11, i64* @sl, align 8
  %12 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %13 = atomicrmw add i64* %12, i64 11 monotonic
  store i64 %13, i64* @ul, align 8
  %14 = atomicrmw sub i8* @sc, i8 11 monotonic
  store i8 %14, i8* @sc, align 1
  %15 = atomicrmw sub i8* @uc, i8 11 monotonic
  store i8 %15, i8* @uc, align 1
  %16 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %17 = atomicrmw sub i16* %16, i16 11 monotonic
  store i16 %17, i16* @ss, align 2
  %18 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %19 = atomicrmw sub i16* %18, i16 11 monotonic
  store i16 %19, i16* @us, align 2
  %20 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %21 = atomicrmw sub i32* %20, i32 11 monotonic
  store i32 %21, i32* @si, align 4
  %22 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %23 = atomicrmw sub i32* %22, i32 11 monotonic
  store i32 %23, i32* @ui, align 4
  %24 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %25 = atomicrmw sub i64* %24, i64 11 monotonic
  store i64 %25, i64* @sl, align 8
  %26 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %27 = atomicrmw sub i64* %26, i64 11 monotonic
  store i64 %27, i64* @ul, align 8
  %28 = atomicrmw or i8* @sc, i8 11 monotonic
  store i8 %28, i8* @sc, align 1
  %29 = atomicrmw or i8* @uc, i8 11 monotonic
  store i8 %29, i8* @uc, align 1
  %30 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %31 = atomicrmw or i16* %30, i16 11 monotonic
  store i16 %31, i16* @ss, align 2
  %32 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %33 = atomicrmw or i16* %32, i16 11 monotonic
  store i16 %33, i16* @us, align 2
  %34 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %35 = atomicrmw or i32* %34, i32 11 monotonic
  store i32 %35, i32* @si, align 4
  %36 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %37 = atomicrmw or i32* %36, i32 11 monotonic
  store i32 %37, i32* @ui, align 4
  %38 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %39 = atomicrmw or i64* %38, i64 11 monotonic
  store i64 %39, i64* @sl, align 8
  %40 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %41 = atomicrmw or i64* %40, i64 11 monotonic
  store i64 %41, i64* @ul, align 8
  %42 = atomicrmw xor i8* @sc, i8 11 monotonic
  store i8 %42, i8* @sc, align 1
  %43 = atomicrmw xor i8* @uc, i8 11 monotonic
  store i8 %43, i8* @uc, align 1
  %44 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %45 = atomicrmw xor i16* %44, i16 11 monotonic
  store i16 %45, i16* @ss, align 2
  %46 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %47 = atomicrmw xor i16* %46, i16 11 monotonic
  store i16 %47, i16* @us, align 2
  %48 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %49 = atomicrmw xor i32* %48, i32 11 monotonic
  store i32 %49, i32* @si, align 4
  %50 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %51 = atomicrmw xor i32* %50, i32 11 monotonic
  store i32 %51, i32* @ui, align 4
  %52 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %53 = atomicrmw xor i64* %52, i64 11 monotonic
  store i64 %53, i64* @sl, align 8
  %54 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %55 = atomicrmw xor i64* %54, i64 11 monotonic
  store i64 %55, i64* @ul, align 8
  %56 = atomicrmw and i8* @sc, i8 11 monotonic
  store i8 %56, i8* @sc, align 1
  %57 = atomicrmw and i8* @uc, i8 11 monotonic
  store i8 %57, i8* @uc, align 1
  %58 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %59 = atomicrmw and i16* %58, i16 11 monotonic
  store i16 %59, i16* @ss, align 2
  %60 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %61 = atomicrmw and i16* %60, i16 11 monotonic
  store i16 %61, i16* @us, align 2
  %62 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %63 = atomicrmw and i32* %62, i32 11 monotonic
  store i32 %63, i32* @si, align 4
  %64 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %65 = atomicrmw and i32* %64, i32 11 monotonic
  store i32 %65, i32* @ui, align 4
  %66 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %67 = atomicrmw and i64* %66, i64 11 monotonic
  store i64 %67, i64* @sl, align 8
  %68 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %69 = atomicrmw and i64* %68, i64 11 monotonic
  store i64 %69, i64* @ul, align 8
  %70 = atomicrmw nand i8* @sc, i8 11 monotonic
  store i8 %70, i8* @sc, align 1
  %71 = atomicrmw nand i8* @uc, i8 11 monotonic
  store i8 %71, i8* @uc, align 1
  %72 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %73 = atomicrmw nand i16* %72, i16 11 monotonic
  store i16 %73, i16* @ss, align 2
  %74 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %75 = atomicrmw nand i16* %74, i16 11 monotonic
  store i16 %75, i16* @us, align 2
  %76 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %77 = atomicrmw nand i32* %76, i32 11 monotonic
  store i32 %77, i32* @si, align 4
  %78 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %79 = atomicrmw nand i32* %78, i32 11 monotonic
  store i32 %79, i32* @ui, align 4
  %80 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %81 = atomicrmw nand i64* %80, i64 11 monotonic
  store i64 %81, i64* @sl, align 8
  %82 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %83 = atomicrmw nand i64* %82, i64 11 monotonic
  store i64 %83, i64* @ul, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_op_and_fetch() nounwind {
entry:
  %0 = load i8, i8* @uc, align 1
  %1 = atomicrmw add i8* @sc, i8 %0 monotonic
  %2 = add i8 %1, %0
  store i8 %2, i8* @sc, align 1
  %3 = load i8, i8* @uc, align 1
  %4 = atomicrmw add i8* @uc, i8 %3 monotonic
  %5 = add i8 %4, %3
  store i8 %5, i8* @uc, align 1
  %6 = load i8, i8* @uc, align 1
  %7 = zext i8 %6 to i16
  %8 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %9 = atomicrmw add i16* %8, i16 %7 monotonic
  %10 = add i16 %9, %7
  store i16 %10, i16* @ss, align 2
  %11 = load i8, i8* @uc, align 1
  %12 = zext i8 %11 to i16
  %13 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %14 = atomicrmw add i16* %13, i16 %12 monotonic
  %15 = add i16 %14, %12
  store i16 %15, i16* @us, align 2
  %16 = load i8, i8* @uc, align 1
  %17 = zext i8 %16 to i32
  %18 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %19 = atomicrmw add i32* %18, i32 %17 monotonic
  %20 = add i32 %19, %17
  store i32 %20, i32* @si, align 4
  %21 = load i8, i8* @uc, align 1
  %22 = zext i8 %21 to i32
  %23 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %24 = atomicrmw add i32* %23, i32 %22 monotonic
  %25 = add i32 %24, %22
  store i32 %25, i32* @ui, align 4
  %26 = load i8, i8* @uc, align 1
  %27 = zext i8 %26 to i64
  %28 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %29 = atomicrmw add i64* %28, i64 %27 monotonic
  %30 = add i64 %29, %27
  store i64 %30, i64* @sl, align 8
  %31 = load i8, i8* @uc, align 1
  %32 = zext i8 %31 to i64
  %33 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %34 = atomicrmw add i64* %33, i64 %32 monotonic
  %35 = add i64 %34, %32
  store i64 %35, i64* @ul, align 8
  %36 = load i8, i8* @uc, align 1
  %37 = atomicrmw sub i8* @sc, i8 %36 monotonic
  %38 = sub i8 %37, %36
  store i8 %38, i8* @sc, align 1
  %39 = load i8, i8* @uc, align 1
  %40 = atomicrmw sub i8* @uc, i8 %39 monotonic
  %41 = sub i8 %40, %39
  store i8 %41, i8* @uc, align 1
  %42 = load i8, i8* @uc, align 1
  %43 = zext i8 %42 to i16
  %44 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %45 = atomicrmw sub i16* %44, i16 %43 monotonic
  %46 = sub i16 %45, %43
  store i16 %46, i16* @ss, align 2
  %47 = load i8, i8* @uc, align 1
  %48 = zext i8 %47 to i16
  %49 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %50 = atomicrmw sub i16* %49, i16 %48 monotonic
  %51 = sub i16 %50, %48
  store i16 %51, i16* @us, align 2
  %52 = load i8, i8* @uc, align 1
  %53 = zext i8 %52 to i32
  %54 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %55 = atomicrmw sub i32* %54, i32 %53 monotonic
  %56 = sub i32 %55, %53
  store i32 %56, i32* @si, align 4
  %57 = load i8, i8* @uc, align 1
  %58 = zext i8 %57 to i32
  %59 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %60 = atomicrmw sub i32* %59, i32 %58 monotonic
  %61 = sub i32 %60, %58
  store i32 %61, i32* @ui, align 4
  %62 = load i8, i8* @uc, align 1
  %63 = zext i8 %62 to i64
  %64 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %65 = atomicrmw sub i64* %64, i64 %63 monotonic
  %66 = sub i64 %65, %63
  store i64 %66, i64* @sl, align 8
  %67 = load i8, i8* @uc, align 1
  %68 = zext i8 %67 to i64
  %69 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %70 = atomicrmw sub i64* %69, i64 %68 monotonic
  %71 = sub i64 %70, %68
  store i64 %71, i64* @ul, align 8
  %72 = load i8, i8* @uc, align 1
  %73 = atomicrmw or i8* @sc, i8 %72 monotonic
  %74 = or i8 %73, %72
  store i8 %74, i8* @sc, align 1
  %75 = load i8, i8* @uc, align 1
  %76 = atomicrmw or i8* @uc, i8 %75 monotonic
  %77 = or i8 %76, %75
  store i8 %77, i8* @uc, align 1
  %78 = load i8, i8* @uc, align 1
  %79 = zext i8 %78 to i16
  %80 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %81 = atomicrmw or i16* %80, i16 %79 monotonic
  %82 = or i16 %81, %79
  store i16 %82, i16* @ss, align 2
  %83 = load i8, i8* @uc, align 1
  %84 = zext i8 %83 to i16
  %85 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %86 = atomicrmw or i16* %85, i16 %84 monotonic
  %87 = or i16 %86, %84
  store i16 %87, i16* @us, align 2
  %88 = load i8, i8* @uc, align 1
  %89 = zext i8 %88 to i32
  %90 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %91 = atomicrmw or i32* %90, i32 %89 monotonic
  %92 = or i32 %91, %89
  store i32 %92, i32* @si, align 4
  %93 = load i8, i8* @uc, align 1
  %94 = zext i8 %93 to i32
  %95 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %96 = atomicrmw or i32* %95, i32 %94 monotonic
  %97 = or i32 %96, %94
  store i32 %97, i32* @ui, align 4
  %98 = load i8, i8* @uc, align 1
  %99 = zext i8 %98 to i64
  %100 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %101 = atomicrmw or i64* %100, i64 %99 monotonic
  %102 = or i64 %101, %99
  store i64 %102, i64* @sl, align 8
  %103 = load i8, i8* @uc, align 1
  %104 = zext i8 %103 to i64
  %105 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %106 = atomicrmw or i64* %105, i64 %104 monotonic
  %107 = or i64 %106, %104
  store i64 %107, i64* @ul, align 8
  %108 = load i8, i8* @uc, align 1
  %109 = atomicrmw xor i8* @sc, i8 %108 monotonic
  %110 = xor i8 %109, %108
  store i8 %110, i8* @sc, align 1
  %111 = load i8, i8* @uc, align 1
  %112 = atomicrmw xor i8* @uc, i8 %111 monotonic
  %113 = xor i8 %112, %111
  store i8 %113, i8* @uc, align 1
  %114 = load i8, i8* @uc, align 1
  %115 = zext i8 %114 to i16
  %116 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %117 = atomicrmw xor i16* %116, i16 %115 monotonic
  %118 = xor i16 %117, %115
  store i16 %118, i16* @ss, align 2
  %119 = load i8, i8* @uc, align 1
  %120 = zext i8 %119 to i16
  %121 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %122 = atomicrmw xor i16* %121, i16 %120 monotonic
  %123 = xor i16 %122, %120
  store i16 %123, i16* @us, align 2
  %124 = load i8, i8* @uc, align 1
  %125 = zext i8 %124 to i32
  %126 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %127 = atomicrmw xor i32* %126, i32 %125 monotonic
  %128 = xor i32 %127, %125
  store i32 %128, i32* @si, align 4
  %129 = load i8, i8* @uc, align 1
  %130 = zext i8 %129 to i32
  %131 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %132 = atomicrmw xor i32* %131, i32 %130 monotonic
  %133 = xor i32 %132, %130
  store i32 %133, i32* @ui, align 4
  %134 = load i8, i8* @uc, align 1
  %135 = zext i8 %134 to i64
  %136 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %137 = atomicrmw xor i64* %136, i64 %135 monotonic
  %138 = xor i64 %137, %135
  store i64 %138, i64* @sl, align 8
  %139 = load i8, i8* @uc, align 1
  %140 = zext i8 %139 to i64
  %141 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %142 = atomicrmw xor i64* %141, i64 %140 monotonic
  %143 = xor i64 %142, %140
  store i64 %143, i64* @ul, align 8
  %144 = load i8, i8* @uc, align 1
  %145 = atomicrmw and i8* @sc, i8 %144 monotonic
  %146 = and i8 %145, %144
  store i8 %146, i8* @sc, align 1
  %147 = load i8, i8* @uc, align 1
  %148 = atomicrmw and i8* @uc, i8 %147 monotonic
  %149 = and i8 %148, %147
  store i8 %149, i8* @uc, align 1
  %150 = load i8, i8* @uc, align 1
  %151 = zext i8 %150 to i16
  %152 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %153 = atomicrmw and i16* %152, i16 %151 monotonic
  %154 = and i16 %153, %151
  store i16 %154, i16* @ss, align 2
  %155 = load i8, i8* @uc, align 1
  %156 = zext i8 %155 to i16
  %157 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %158 = atomicrmw and i16* %157, i16 %156 monotonic
  %159 = and i16 %158, %156
  store i16 %159, i16* @us, align 2
  %160 = load i8, i8* @uc, align 1
  %161 = zext i8 %160 to i32
  %162 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %163 = atomicrmw and i32* %162, i32 %161 monotonic
  %164 = and i32 %163, %161
  store i32 %164, i32* @si, align 4
  %165 = load i8, i8* @uc, align 1
  %166 = zext i8 %165 to i32
  %167 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %168 = atomicrmw and i32* %167, i32 %166 monotonic
  %169 = and i32 %168, %166
  store i32 %169, i32* @ui, align 4
  %170 = load i8, i8* @uc, align 1
  %171 = zext i8 %170 to i64
  %172 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %173 = atomicrmw and i64* %172, i64 %171 monotonic
  %174 = and i64 %173, %171
  store i64 %174, i64* @sl, align 8
  %175 = load i8, i8* @uc, align 1
  %176 = zext i8 %175 to i64
  %177 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %178 = atomicrmw and i64* %177, i64 %176 monotonic
  %179 = and i64 %178, %176
  store i64 %179, i64* @ul, align 8
  %180 = load i8, i8* @uc, align 1
  %181 = atomicrmw nand i8* @sc, i8 %180 monotonic
  %182 = xor i8 %181, -1
  %183 = and i8 %182, %180
  store i8 %183, i8* @sc, align 1
  %184 = load i8, i8* @uc, align 1
  %185 = atomicrmw nand i8* @uc, i8 %184 monotonic
  %186 = xor i8 %185, -1
  %187 = and i8 %186, %184
  store i8 %187, i8* @uc, align 1
  %188 = load i8, i8* @uc, align 1
  %189 = zext i8 %188 to i16
  %190 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %191 = atomicrmw nand i16* %190, i16 %189 monotonic
  %192 = xor i16 %191, -1
  %193 = and i16 %192, %189
  store i16 %193, i16* @ss, align 2
  %194 = load i8, i8* @uc, align 1
  %195 = zext i8 %194 to i16
  %196 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %197 = atomicrmw nand i16* %196, i16 %195 monotonic
  %198 = xor i16 %197, -1
  %199 = and i16 %198, %195
  store i16 %199, i16* @us, align 2
  %200 = load i8, i8* @uc, align 1
  %201 = zext i8 %200 to i32
  %202 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %203 = atomicrmw nand i32* %202, i32 %201 monotonic
  %204 = xor i32 %203, -1
  %205 = and i32 %204, %201
  store i32 %205, i32* @si, align 4
  %206 = load i8, i8* @uc, align 1
  %207 = zext i8 %206 to i32
  %208 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %209 = atomicrmw nand i32* %208, i32 %207 monotonic
  %210 = xor i32 %209, -1
  %211 = and i32 %210, %207
  store i32 %211, i32* @ui, align 4
  %212 = load i8, i8* @uc, align 1
  %213 = zext i8 %212 to i64
  %214 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %215 = atomicrmw nand i64* %214, i64 %213 monotonic
  %216 = xor i64 %215, -1
  %217 = and i64 %216, %213
  store i64 %217, i64* @sl, align 8
  %218 = load i8, i8* @uc, align 1
  %219 = zext i8 %218 to i64
  %220 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %221 = atomicrmw nand i64* %220, i64 %219 monotonic
  %222 = xor i64 %221, -1
  %223 = and i64 %222, %219
  store i64 %223, i64* @ul, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_compare_and_swap() nounwind {
entry:
  %0 = load i8, i8* @uc, align 1
  %1 = load i8, i8* @sc, align 1
  %2 = cmpxchg i8* @sc, i8 %0, i8 %1 monotonic monotonic
  store i8 %2, i8* @sc, align 1
  %3 = load i8, i8* @uc, align 1
  %4 = load i8, i8* @sc, align 1
  %5 = cmpxchg i8* @uc, i8 %3, i8 %4 monotonic monotonic
  store i8 %5, i8* @uc, align 1
  %6 = load i8, i8* @uc, align 1
  %7 = zext i8 %6 to i16
  %8 = load i8, i8* @sc, align 1
  %9 = sext i8 %8 to i16
  %10 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %11 = cmpxchg i16* %10, i16 %7, i16 %9 monotonic monotonic
  store i16 %11, i16* @ss, align 2
  %12 = load i8, i8* @uc, align 1
  %13 = zext i8 %12 to i16
  %14 = load i8, i8* @sc, align 1
  %15 = sext i8 %14 to i16
  %16 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %17 = cmpxchg i16* %16, i16 %13, i16 %15 monotonic monotonic
  store i16 %17, i16* @us, align 2
  %18 = load i8, i8* @uc, align 1
  %19 = zext i8 %18 to i32
  %20 = load i8, i8* @sc, align 1
  %21 = sext i8 %20 to i32
  %22 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %23 = cmpxchg i32* %22, i32 %19, i32 %21 monotonic monotonic
  store i32 %23, i32* @si, align 4
  %24 = load i8, i8* @uc, align 1
  %25 = zext i8 %24 to i32
  %26 = load i8, i8* @sc, align 1
  %27 = sext i8 %26 to i32
  %28 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %29 = cmpxchg i32* %28, i32 %25, i32 %27 monotonic monotonic
  store i32 %29, i32* @ui, align 4
  %30 = load i8, i8* @uc, align 1
  %31 = zext i8 %30 to i64
  %32 = load i8, i8* @sc, align 1
  %33 = sext i8 %32 to i64
  %34 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %35 = cmpxchg i64* %34, i64 %31, i64 %33 monotonic monotonic
  store i64 %35, i64* @sl, align 8
  %36 = load i8, i8* @uc, align 1
  %37 = zext i8 %36 to i64
  %38 = load i8, i8* @sc, align 1
  %39 = sext i8 %38 to i64
  %40 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %41 = cmpxchg i64* %40, i64 %37, i64 %39 monotonic monotonic
  store i64 %41, i64* @ul, align 8
  %42 = load i8, i8* @uc, align 1
  %43 = load i8, i8* @sc, align 1
  %44 = cmpxchg i8* @sc, i8 %42, i8 %43 monotonic monotonic
  %45 = icmp eq i8 %44, %42
  %46 = zext i1 %45 to i8
  %47 = zext i8 %46 to i32
  store i32 %47, i32* @ui, align 4
  %48 = load i8, i8* @uc, align 1
  %49 = load i8, i8* @sc, align 1
  %50 = cmpxchg i8* @uc, i8 %48, i8 %49 monotonic monotonic
  %51 = icmp eq i8 %50, %48
  %52 = zext i1 %51 to i8
  %53 = zext i8 %52 to i32
  store i32 %53, i32* @ui, align 4
  %54 = load i8, i8* @uc, align 1
  %55 = zext i8 %54 to i16
  %56 = load i8, i8* @sc, align 1
  %57 = sext i8 %56 to i16
  %58 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %59 = cmpxchg i16* %58, i16 %55, i16 %57 monotonic monotonic
  %60 = icmp eq i16 %59, %55
  %61 = zext i1 %60 to i8
  %62 = zext i8 %61 to i32
  store i32 %62, i32* @ui, align 4
  %63 = load i8, i8* @uc, align 1
  %64 = zext i8 %63 to i16
  %65 = load i8, i8* @sc, align 1
  %66 = sext i8 %65 to i16
  %67 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %68 = cmpxchg i16* %67, i16 %64, i16 %66 monotonic monotonic
  %69 = icmp eq i16 %68, %64
  %70 = zext i1 %69 to i8
  %71 = zext i8 %70 to i32
  store i32 %71, i32* @ui, align 4
  %72 = load i8, i8* @uc, align 1
  %73 = zext i8 %72 to i32
  %74 = load i8, i8* @sc, align 1
  %75 = sext i8 %74 to i32
  %76 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %77 = cmpxchg i32* %76, i32 %73, i32 %75 monotonic monotonic
  %78 = icmp eq i32 %77, %73
  %79 = zext i1 %78 to i8
  %80 = zext i8 %79 to i32
  store i32 %80, i32* @ui, align 4
  %81 = load i8, i8* @uc, align 1
  %82 = zext i8 %81 to i32
  %83 = load i8, i8* @sc, align 1
  %84 = sext i8 %83 to i32
  %85 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %86 = cmpxchg i32* %85, i32 %82, i32 %84 monotonic monotonic
  %87 = icmp eq i32 %86, %82
  %88 = zext i1 %87 to i8
  %89 = zext i8 %88 to i32
  store i32 %89, i32* @ui, align 4
  %90 = load i8, i8* @uc, align 1
  %91 = zext i8 %90 to i64
  %92 = load i8, i8* @sc, align 1
  %93 = sext i8 %92 to i64
  %94 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %95 = cmpxchg i64* %94, i64 %91, i64 %93 monotonic monotonic
  %96 = icmp eq i64 %95, %91
  %97 = zext i1 %96 to i8
  %98 = zext i8 %97 to i32
  store i32 %98, i32* @ui, align 4
  %99 = load i8, i8* @uc, align 1
  %100 = zext i8 %99 to i64
  %101 = load i8, i8* @sc, align 1
  %102 = sext i8 %101 to i64
  %103 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %104 = cmpxchg i64* %103, i64 %100, i64 %102 monotonic monotonic
  %105 = icmp eq i64 %104, %100
  %106 = zext i1 %105 to i8
  %107 = zext i8 %106 to i32
  store i32 %107, i32* @ui, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_lock() nounwind {
entry:
  %0 = atomicrmw xchg i8* @sc, i8 1 monotonic
  store i8 %0, i8* @sc, align 1
  %1 = atomicrmw xchg i8* @uc, i8 1 monotonic
  store i8 %1, i8* @uc, align 1
  %2 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %3 = atomicrmw xchg i16* %2, i16 1 monotonic
  store i16 %3, i16* @ss, align 2
  %4 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %5 = atomicrmw xchg i16* %4, i16 1 monotonic
  store i16 %5, i16* @us, align 2
  %6 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %7 = atomicrmw xchg i32* %6, i32 1 monotonic
  store i32 %7, i32* @si, align 4
  %8 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %9 = atomicrmw xchg i32* %8, i32 1 monotonic
  store i32 %9, i32* @ui, align 4
  %10 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %11 = atomicrmw xchg i64* %10, i64 1 monotonic
  store i64 %11, i64* @sl, align 8
  %12 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %13 = atomicrmw xchg i64* %12, i64 1 monotonic
  store i64 %13, i64* @ul, align 8
  fence seq_cst
  store volatile i8 0, i8* @sc, align 1
  store volatile i8 0, i8* @uc, align 1
  %14 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  store volatile i16 0, i16* %14, align 2
  %15 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  store volatile i16 0, i16* %15, align 2
  %16 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  store volatile i32 0, i32* %16, align 4
  %17 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  store volatile i32 0, i32* %17, align 4
  %18 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  store volatile i64 0, i64* %18, align 8
  %19 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  store volatile i64 0, i64* %19, align 8
  %20 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  store volatile i64 0, i64* %20, align 8
  %21 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  store volatile i64 0, i64* %21, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}
