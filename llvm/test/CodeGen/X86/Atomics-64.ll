; RUN: llc < %s -march=x86-64 > %t
; RUN: llc < %s -march=x86 > %t
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"

@sc = common global i8 0
@uc = common global i8 0
@ss = common global i16 0
@us = common global i16 0
@si = common global i32 0
@ui = common global i32 0
@sl = common global i64 0
@ul = common global i64 0
@sll = common global i64 0
@ull = common global i64 0

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
  %14 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %15 = atomicrmw add i64* %14, i64 1 monotonic
  %16 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %17 = atomicrmw add i64* %16, i64 1 monotonic
  %18 = atomicrmw sub i8* @sc, i8 1 monotonic
  %19 = atomicrmw sub i8* @uc, i8 1 monotonic
  %20 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %21 = atomicrmw sub i16* %20, i16 1 monotonic
  %22 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %23 = atomicrmw sub i16* %22, i16 1 monotonic
  %24 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %25 = atomicrmw sub i32* %24, i32 1 monotonic
  %26 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %27 = atomicrmw sub i32* %26, i32 1 monotonic
  %28 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %29 = atomicrmw sub i64* %28, i64 1 monotonic
  %30 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %31 = atomicrmw sub i64* %30, i64 1 monotonic
  %32 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %33 = atomicrmw sub i64* %32, i64 1 monotonic
  %34 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %35 = atomicrmw sub i64* %34, i64 1 monotonic
  %36 = atomicrmw or i8* @sc, i8 1 monotonic
  %37 = atomicrmw or i8* @uc, i8 1 monotonic
  %38 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %39 = atomicrmw or i16* %38, i16 1 monotonic
  %40 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %41 = atomicrmw or i16* %40, i16 1 monotonic
  %42 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %43 = atomicrmw or i32* %42, i32 1 monotonic
  %44 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %45 = atomicrmw or i32* %44, i32 1 monotonic
  %46 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %47 = atomicrmw or i64* %46, i64 1 monotonic
  %48 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %49 = atomicrmw or i64* %48, i64 1 monotonic
  %50 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %51 = atomicrmw or i64* %50, i64 1 monotonic
  %52 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %53 = atomicrmw or i64* %52, i64 1 monotonic
  %54 = atomicrmw xor i8* @sc, i8 1 monotonic
  %55 = atomicrmw xor i8* @uc, i8 1 monotonic
  %56 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %57 = atomicrmw xor i16* %56, i16 1 monotonic
  %58 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %59 = atomicrmw xor i16* %58, i16 1 monotonic
  %60 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %61 = atomicrmw xor i32* %60, i32 1 monotonic
  %62 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %63 = atomicrmw xor i32* %62, i32 1 monotonic
  %64 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %65 = atomicrmw xor i64* %64, i64 1 monotonic
  %66 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %67 = atomicrmw xor i64* %66, i64 1 monotonic
  %68 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %69 = atomicrmw xor i64* %68, i64 1 monotonic
  %70 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %71 = atomicrmw xor i64* %70, i64 1 monotonic
  %72 = atomicrmw and i8* @sc, i8 1 monotonic
  %73 = atomicrmw and i8* @uc, i8 1 monotonic
  %74 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %75 = atomicrmw and i16* %74, i16 1 monotonic
  %76 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %77 = atomicrmw and i16* %76, i16 1 monotonic
  %78 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %79 = atomicrmw and i32* %78, i32 1 monotonic
  %80 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %81 = atomicrmw and i32* %80, i32 1 monotonic
  %82 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %83 = atomicrmw and i64* %82, i64 1 monotonic
  %84 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %85 = atomicrmw and i64* %84, i64 1 monotonic
  %86 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %87 = atomicrmw and i64* %86, i64 1 monotonic
  %88 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %89 = atomicrmw and i64* %88, i64 1 monotonic
  %90 = atomicrmw nand i8* @sc, i8 1 monotonic
  %91 = atomicrmw nand i8* @uc, i8 1 monotonic
  %92 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %93 = atomicrmw nand i16* %92, i16 1 monotonic
  %94 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %95 = atomicrmw nand i16* %94, i16 1 monotonic
  %96 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %97 = atomicrmw nand i32* %96, i32 1 monotonic
  %98 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %99 = atomicrmw nand i32* %98, i32 1 monotonic
  %100 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %101 = atomicrmw nand i64* %100, i64 1 monotonic
  %102 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %103 = atomicrmw nand i64* %102, i64 1 monotonic
  %104 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %105 = atomicrmw nand i64* %104, i64 1 monotonic
  %106 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %107 = atomicrmw nand i64* %106, i64 1 monotonic
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
  %14 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %15 = atomicrmw add i64* %14, i64 11 monotonic
  store i64 %15, i64* @sll, align 8
  %16 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %17 = atomicrmw add i64* %16, i64 11 monotonic
  store i64 %17, i64* @ull, align 8
  %18 = atomicrmw sub i8* @sc, i8 11 monotonic
  store i8 %18, i8* @sc, align 1
  %19 = atomicrmw sub i8* @uc, i8 11 monotonic
  store i8 %19, i8* @uc, align 1
  %20 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %21 = atomicrmw sub i16* %20, i16 11 monotonic
  store i16 %21, i16* @ss, align 2
  %22 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %23 = atomicrmw sub i16* %22, i16 11 monotonic
  store i16 %23, i16* @us, align 2
  %24 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %25 = atomicrmw sub i32* %24, i32 11 monotonic
  store i32 %25, i32* @si, align 4
  %26 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %27 = atomicrmw sub i32* %26, i32 11 monotonic
  store i32 %27, i32* @ui, align 4
  %28 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %29 = atomicrmw sub i64* %28, i64 11 monotonic
  store i64 %29, i64* @sl, align 8
  %30 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %31 = atomicrmw sub i64* %30, i64 11 monotonic
  store i64 %31, i64* @ul, align 8
  %32 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %33 = atomicrmw sub i64* %32, i64 11 monotonic
  store i64 %33, i64* @sll, align 8
  %34 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %35 = atomicrmw sub i64* %34, i64 11 monotonic
  store i64 %35, i64* @ull, align 8
  %36 = atomicrmw or i8* @sc, i8 11 monotonic
  store i8 %36, i8* @sc, align 1
  %37 = atomicrmw or i8* @uc, i8 11 monotonic
  store i8 %37, i8* @uc, align 1
  %38 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %39 = atomicrmw or i16* %38, i16 11 monotonic
  store i16 %39, i16* @ss, align 2
  %40 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %41 = atomicrmw or i16* %40, i16 11 monotonic
  store i16 %41, i16* @us, align 2
  %42 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %43 = atomicrmw or i32* %42, i32 11 monotonic
  store i32 %43, i32* @si, align 4
  %44 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %45 = atomicrmw or i32* %44, i32 11 monotonic
  store i32 %45, i32* @ui, align 4
  %46 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %47 = atomicrmw or i64* %46, i64 11 monotonic
  store i64 %47, i64* @sl, align 8
  %48 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %49 = atomicrmw or i64* %48, i64 11 monotonic
  store i64 %49, i64* @ul, align 8
  %50 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %51 = atomicrmw or i64* %50, i64 11 monotonic
  store i64 %51, i64* @sll, align 8
  %52 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %53 = atomicrmw or i64* %52, i64 11 monotonic
  store i64 %53, i64* @ull, align 8
  %54 = atomicrmw xor i8* @sc, i8 11 monotonic
  store i8 %54, i8* @sc, align 1
  %55 = atomicrmw xor i8* @uc, i8 11 monotonic
  store i8 %55, i8* @uc, align 1
  %56 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %57 = atomicrmw xor i16* %56, i16 11 monotonic
  store i16 %57, i16* @ss, align 2
  %58 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %59 = atomicrmw xor i16* %58, i16 11 monotonic
  store i16 %59, i16* @us, align 2
  %60 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %61 = atomicrmw xor i32* %60, i32 11 monotonic
  store i32 %61, i32* @si, align 4
  %62 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %63 = atomicrmw xor i32* %62, i32 11 monotonic
  store i32 %63, i32* @ui, align 4
  %64 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %65 = atomicrmw xor i64* %64, i64 11 monotonic
  store i64 %65, i64* @sl, align 8
  %66 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %67 = atomicrmw xor i64* %66, i64 11 monotonic
  store i64 %67, i64* @ul, align 8
  %68 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %69 = atomicrmw xor i64* %68, i64 11 monotonic
  store i64 %69, i64* @sll, align 8
  %70 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %71 = atomicrmw xor i64* %70, i64 11 monotonic
  store i64 %71, i64* @ull, align 8
  %72 = atomicrmw and i8* @sc, i8 11 monotonic
  store i8 %72, i8* @sc, align 1
  %73 = atomicrmw and i8* @uc, i8 11 monotonic
  store i8 %73, i8* @uc, align 1
  %74 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %75 = atomicrmw and i16* %74, i16 11 monotonic
  store i16 %75, i16* @ss, align 2
  %76 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %77 = atomicrmw and i16* %76, i16 11 monotonic
  store i16 %77, i16* @us, align 2
  %78 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %79 = atomicrmw and i32* %78, i32 11 monotonic
  store i32 %79, i32* @si, align 4
  %80 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %81 = atomicrmw and i32* %80, i32 11 monotonic
  store i32 %81, i32* @ui, align 4
  %82 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %83 = atomicrmw and i64* %82, i64 11 monotonic
  store i64 %83, i64* @sl, align 8
  %84 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %85 = atomicrmw and i64* %84, i64 11 monotonic
  store i64 %85, i64* @ul, align 8
  %86 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %87 = atomicrmw and i64* %86, i64 11 monotonic
  store i64 %87, i64* @sll, align 8
  %88 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %89 = atomicrmw and i64* %88, i64 11 monotonic
  store i64 %89, i64* @ull, align 8
  %90 = atomicrmw nand i8* @sc, i8 11 monotonic
  store i8 %90, i8* @sc, align 1
  %91 = atomicrmw nand i8* @uc, i8 11 monotonic
  store i8 %91, i8* @uc, align 1
  %92 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %93 = atomicrmw nand i16* %92, i16 11 monotonic
  store i16 %93, i16* @ss, align 2
  %94 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %95 = atomicrmw nand i16* %94, i16 11 monotonic
  store i16 %95, i16* @us, align 2
  %96 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %97 = atomicrmw nand i32* %96, i32 11 monotonic
  store i32 %97, i32* @si, align 4
  %98 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %99 = atomicrmw nand i32* %98, i32 11 monotonic
  store i32 %99, i32* @ui, align 4
  %100 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %101 = atomicrmw nand i64* %100, i64 11 monotonic
  store i64 %101, i64* @sl, align 8
  %102 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %103 = atomicrmw nand i64* %102, i64 11 monotonic
  store i64 %103, i64* @ul, align 8
  %104 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %105 = atomicrmw nand i64* %104, i64 11 monotonic
  store i64 %105, i64* @sll, align 8
  %106 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %107 = atomicrmw nand i64* %106, i64 11 monotonic
  store i64 %107, i64* @ull, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_op_and_fetch() nounwind {
entry:
  %0 = load i8* @uc, align 1
  %1 = zext i8 %0 to i32
  %2 = trunc i32 %1 to i8
  %3 = atomicrmw add i8* @sc, i8 %2 monotonic
  %4 = add i8 %3, %2
  store i8 %4, i8* @sc, align 1
  %5 = load i8* @uc, align 1
  %6 = zext i8 %5 to i32
  %7 = trunc i32 %6 to i8
  %8 = atomicrmw add i8* @uc, i8 %7 monotonic
  %9 = add i8 %8, %7
  store i8 %9, i8* @uc, align 1
  %10 = load i8* @uc, align 1
  %11 = zext i8 %10 to i32
  %12 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %13 = trunc i32 %11 to i16
  %14 = atomicrmw add i16* %12, i16 %13 monotonic
  %15 = add i16 %14, %13
  store i16 %15, i16* @ss, align 2
  %16 = load i8* @uc, align 1
  %17 = zext i8 %16 to i32
  %18 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %19 = trunc i32 %17 to i16
  %20 = atomicrmw add i16* %18, i16 %19 monotonic
  %21 = add i16 %20, %19
  store i16 %21, i16* @us, align 2
  %22 = load i8* @uc, align 1
  %23 = zext i8 %22 to i32
  %24 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %25 = atomicrmw add i32* %24, i32 %23 monotonic
  %26 = add i32 %25, %23
  store i32 %26, i32* @si, align 4
  %27 = load i8* @uc, align 1
  %28 = zext i8 %27 to i32
  %29 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %30 = atomicrmw add i32* %29, i32 %28 monotonic
  %31 = add i32 %30, %28
  store i32 %31, i32* @ui, align 4
  %32 = load i8* @uc, align 1
  %33 = zext i8 %32 to i64
  %34 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %35 = atomicrmw add i64* %34, i64 %33 monotonic
  %36 = add i64 %35, %33
  store i64 %36, i64* @sl, align 8
  %37 = load i8* @uc, align 1
  %38 = zext i8 %37 to i64
  %39 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %40 = atomicrmw add i64* %39, i64 %38 monotonic
  %41 = add i64 %40, %38
  store i64 %41, i64* @ul, align 8
  %42 = load i8* @uc, align 1
  %43 = zext i8 %42 to i64
  %44 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %45 = atomicrmw add i64* %44, i64 %43 monotonic
  %46 = add i64 %45, %43
  store i64 %46, i64* @sll, align 8
  %47 = load i8* @uc, align 1
  %48 = zext i8 %47 to i64
  %49 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %50 = atomicrmw add i64* %49, i64 %48 monotonic
  %51 = add i64 %50, %48
  store i64 %51, i64* @ull, align 8
  %52 = load i8* @uc, align 1
  %53 = zext i8 %52 to i32
  %54 = trunc i32 %53 to i8
  %55 = atomicrmw sub i8* @sc, i8 %54 monotonic
  %56 = sub i8 %55, %54
  store i8 %56, i8* @sc, align 1
  %57 = load i8* @uc, align 1
  %58 = zext i8 %57 to i32
  %59 = trunc i32 %58 to i8
  %60 = atomicrmw sub i8* @uc, i8 %59 monotonic
  %61 = sub i8 %60, %59
  store i8 %61, i8* @uc, align 1
  %62 = load i8* @uc, align 1
  %63 = zext i8 %62 to i32
  %64 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %65 = trunc i32 %63 to i16
  %66 = atomicrmw sub i16* %64, i16 %65 monotonic
  %67 = sub i16 %66, %65
  store i16 %67, i16* @ss, align 2
  %68 = load i8* @uc, align 1
  %69 = zext i8 %68 to i32
  %70 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %71 = trunc i32 %69 to i16
  %72 = atomicrmw sub i16* %70, i16 %71 monotonic
  %73 = sub i16 %72, %71
  store i16 %73, i16* @us, align 2
  %74 = load i8* @uc, align 1
  %75 = zext i8 %74 to i32
  %76 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %77 = atomicrmw sub i32* %76, i32 %75 monotonic
  %78 = sub i32 %77, %75
  store i32 %78, i32* @si, align 4
  %79 = load i8* @uc, align 1
  %80 = zext i8 %79 to i32
  %81 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %82 = atomicrmw sub i32* %81, i32 %80 monotonic
  %83 = sub i32 %82, %80
  store i32 %83, i32* @ui, align 4
  %84 = load i8* @uc, align 1
  %85 = zext i8 %84 to i64
  %86 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %87 = atomicrmw sub i64* %86, i64 %85 monotonic
  %88 = sub i64 %87, %85
  store i64 %88, i64* @sl, align 8
  %89 = load i8* @uc, align 1
  %90 = zext i8 %89 to i64
  %91 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %92 = atomicrmw sub i64* %91, i64 %90 monotonic
  %93 = sub i64 %92, %90
  store i64 %93, i64* @ul, align 8
  %94 = load i8* @uc, align 1
  %95 = zext i8 %94 to i64
  %96 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %97 = atomicrmw sub i64* %96, i64 %95 monotonic
  %98 = sub i64 %97, %95
  store i64 %98, i64* @sll, align 8
  %99 = load i8* @uc, align 1
  %100 = zext i8 %99 to i64
  %101 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %102 = atomicrmw sub i64* %101, i64 %100 monotonic
  %103 = sub i64 %102, %100
  store i64 %103, i64* @ull, align 8
  %104 = load i8* @uc, align 1
  %105 = zext i8 %104 to i32
  %106 = trunc i32 %105 to i8
  %107 = atomicrmw or i8* @sc, i8 %106 monotonic
  %108 = or i8 %107, %106
  store i8 %108, i8* @sc, align 1
  %109 = load i8* @uc, align 1
  %110 = zext i8 %109 to i32
  %111 = trunc i32 %110 to i8
  %112 = atomicrmw or i8* @uc, i8 %111 monotonic
  %113 = or i8 %112, %111
  store i8 %113, i8* @uc, align 1
  %114 = load i8* @uc, align 1
  %115 = zext i8 %114 to i32
  %116 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %117 = trunc i32 %115 to i16
  %118 = atomicrmw or i16* %116, i16 %117 monotonic
  %119 = or i16 %118, %117
  store i16 %119, i16* @ss, align 2
  %120 = load i8* @uc, align 1
  %121 = zext i8 %120 to i32
  %122 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %123 = trunc i32 %121 to i16
  %124 = atomicrmw or i16* %122, i16 %123 monotonic
  %125 = or i16 %124, %123
  store i16 %125, i16* @us, align 2
  %126 = load i8* @uc, align 1
  %127 = zext i8 %126 to i32
  %128 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %129 = atomicrmw or i32* %128, i32 %127 monotonic
  %130 = or i32 %129, %127
  store i32 %130, i32* @si, align 4
  %131 = load i8* @uc, align 1
  %132 = zext i8 %131 to i32
  %133 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %134 = atomicrmw or i32* %133, i32 %132 monotonic
  %135 = or i32 %134, %132
  store i32 %135, i32* @ui, align 4
  %136 = load i8* @uc, align 1
  %137 = zext i8 %136 to i64
  %138 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %139 = atomicrmw or i64* %138, i64 %137 monotonic
  %140 = or i64 %139, %137
  store i64 %140, i64* @sl, align 8
  %141 = load i8* @uc, align 1
  %142 = zext i8 %141 to i64
  %143 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %144 = atomicrmw or i64* %143, i64 %142 monotonic
  %145 = or i64 %144, %142
  store i64 %145, i64* @ul, align 8
  %146 = load i8* @uc, align 1
  %147 = zext i8 %146 to i64
  %148 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %149 = atomicrmw or i64* %148, i64 %147 monotonic
  %150 = or i64 %149, %147
  store i64 %150, i64* @sll, align 8
  %151 = load i8* @uc, align 1
  %152 = zext i8 %151 to i64
  %153 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %154 = atomicrmw or i64* %153, i64 %152 monotonic
  %155 = or i64 %154, %152
  store i64 %155, i64* @ull, align 8
  %156 = load i8* @uc, align 1
  %157 = zext i8 %156 to i32
  %158 = trunc i32 %157 to i8
  %159 = atomicrmw xor i8* @sc, i8 %158 monotonic
  %160 = xor i8 %159, %158
  store i8 %160, i8* @sc, align 1
  %161 = load i8* @uc, align 1
  %162 = zext i8 %161 to i32
  %163 = trunc i32 %162 to i8
  %164 = atomicrmw xor i8* @uc, i8 %163 monotonic
  %165 = xor i8 %164, %163
  store i8 %165, i8* @uc, align 1
  %166 = load i8* @uc, align 1
  %167 = zext i8 %166 to i32
  %168 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %169 = trunc i32 %167 to i16
  %170 = atomicrmw xor i16* %168, i16 %169 monotonic
  %171 = xor i16 %170, %169
  store i16 %171, i16* @ss, align 2
  %172 = load i8* @uc, align 1
  %173 = zext i8 %172 to i32
  %174 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %175 = trunc i32 %173 to i16
  %176 = atomicrmw xor i16* %174, i16 %175 monotonic
  %177 = xor i16 %176, %175
  store i16 %177, i16* @us, align 2
  %178 = load i8* @uc, align 1
  %179 = zext i8 %178 to i32
  %180 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %181 = atomicrmw xor i32* %180, i32 %179 monotonic
  %182 = xor i32 %181, %179
  store i32 %182, i32* @si, align 4
  %183 = load i8* @uc, align 1
  %184 = zext i8 %183 to i32
  %185 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %186 = atomicrmw xor i32* %185, i32 %184 monotonic
  %187 = xor i32 %186, %184
  store i32 %187, i32* @ui, align 4
  %188 = load i8* @uc, align 1
  %189 = zext i8 %188 to i64
  %190 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %191 = atomicrmw xor i64* %190, i64 %189 monotonic
  %192 = xor i64 %191, %189
  store i64 %192, i64* @sl, align 8
  %193 = load i8* @uc, align 1
  %194 = zext i8 %193 to i64
  %195 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %196 = atomicrmw xor i64* %195, i64 %194 monotonic
  %197 = xor i64 %196, %194
  store i64 %197, i64* @ul, align 8
  %198 = load i8* @uc, align 1
  %199 = zext i8 %198 to i64
  %200 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %201 = atomicrmw xor i64* %200, i64 %199 monotonic
  %202 = xor i64 %201, %199
  store i64 %202, i64* @sll, align 8
  %203 = load i8* @uc, align 1
  %204 = zext i8 %203 to i64
  %205 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %206 = atomicrmw xor i64* %205, i64 %204 monotonic
  %207 = xor i64 %206, %204
  store i64 %207, i64* @ull, align 8
  %208 = load i8* @uc, align 1
  %209 = zext i8 %208 to i32
  %210 = trunc i32 %209 to i8
  %211 = atomicrmw and i8* @sc, i8 %210 monotonic
  %212 = and i8 %211, %210
  store i8 %212, i8* @sc, align 1
  %213 = load i8* @uc, align 1
  %214 = zext i8 %213 to i32
  %215 = trunc i32 %214 to i8
  %216 = atomicrmw and i8* @uc, i8 %215 monotonic
  %217 = and i8 %216, %215
  store i8 %217, i8* @uc, align 1
  %218 = load i8* @uc, align 1
  %219 = zext i8 %218 to i32
  %220 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %221 = trunc i32 %219 to i16
  %222 = atomicrmw and i16* %220, i16 %221 monotonic
  %223 = and i16 %222, %221
  store i16 %223, i16* @ss, align 2
  %224 = load i8* @uc, align 1
  %225 = zext i8 %224 to i32
  %226 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %227 = trunc i32 %225 to i16
  %228 = atomicrmw and i16* %226, i16 %227 monotonic
  %229 = and i16 %228, %227
  store i16 %229, i16* @us, align 2
  %230 = load i8* @uc, align 1
  %231 = zext i8 %230 to i32
  %232 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %233 = atomicrmw and i32* %232, i32 %231 monotonic
  %234 = and i32 %233, %231
  store i32 %234, i32* @si, align 4
  %235 = load i8* @uc, align 1
  %236 = zext i8 %235 to i32
  %237 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %238 = atomicrmw and i32* %237, i32 %236 monotonic
  %239 = and i32 %238, %236
  store i32 %239, i32* @ui, align 4
  %240 = load i8* @uc, align 1
  %241 = zext i8 %240 to i64
  %242 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %243 = atomicrmw and i64* %242, i64 %241 monotonic
  %244 = and i64 %243, %241
  store i64 %244, i64* @sl, align 8
  %245 = load i8* @uc, align 1
  %246 = zext i8 %245 to i64
  %247 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %248 = atomicrmw and i64* %247, i64 %246 monotonic
  %249 = and i64 %248, %246
  store i64 %249, i64* @ul, align 8
  %250 = load i8* @uc, align 1
  %251 = zext i8 %250 to i64
  %252 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %253 = atomicrmw and i64* %252, i64 %251 monotonic
  %254 = and i64 %253, %251
  store i64 %254, i64* @sll, align 8
  %255 = load i8* @uc, align 1
  %256 = zext i8 %255 to i64
  %257 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %258 = atomicrmw and i64* %257, i64 %256 monotonic
  %259 = and i64 %258, %256
  store i64 %259, i64* @ull, align 8
  %260 = load i8* @uc, align 1
  %261 = zext i8 %260 to i32
  %262 = trunc i32 %261 to i8
  %263 = atomicrmw nand i8* @sc, i8 %262 monotonic
  %264 = xor i8 %263, -1
  %265 = and i8 %264, %262
  store i8 %265, i8* @sc, align 1
  %266 = load i8* @uc, align 1
  %267 = zext i8 %266 to i32
  %268 = trunc i32 %267 to i8
  %269 = atomicrmw nand i8* @uc, i8 %268 monotonic
  %270 = xor i8 %269, -1
  %271 = and i8 %270, %268
  store i8 %271, i8* @uc, align 1
  %272 = load i8* @uc, align 1
  %273 = zext i8 %272 to i32
  %274 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %275 = trunc i32 %273 to i16
  %276 = atomicrmw nand i16* %274, i16 %275 monotonic
  %277 = xor i16 %276, -1
  %278 = and i16 %277, %275
  store i16 %278, i16* @ss, align 2
  %279 = load i8* @uc, align 1
  %280 = zext i8 %279 to i32
  %281 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %282 = trunc i32 %280 to i16
  %283 = atomicrmw nand i16* %281, i16 %282 monotonic
  %284 = xor i16 %283, -1
  %285 = and i16 %284, %282
  store i16 %285, i16* @us, align 2
  %286 = load i8* @uc, align 1
  %287 = zext i8 %286 to i32
  %288 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %289 = atomicrmw nand i32* %288, i32 %287 monotonic
  %290 = xor i32 %289, -1
  %291 = and i32 %290, %287
  store i32 %291, i32* @si, align 4
  %292 = load i8* @uc, align 1
  %293 = zext i8 %292 to i32
  %294 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %295 = atomicrmw nand i32* %294, i32 %293 monotonic
  %296 = xor i32 %295, -1
  %297 = and i32 %296, %293
  store i32 %297, i32* @ui, align 4
  %298 = load i8* @uc, align 1
  %299 = zext i8 %298 to i64
  %300 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %301 = atomicrmw nand i64* %300, i64 %299 monotonic
  %302 = xor i64 %301, -1
  %303 = and i64 %302, %299
  store i64 %303, i64* @sl, align 8
  %304 = load i8* @uc, align 1
  %305 = zext i8 %304 to i64
  %306 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %307 = atomicrmw nand i64* %306, i64 %305 monotonic
  %308 = xor i64 %307, -1
  %309 = and i64 %308, %305
  store i64 %309, i64* @ul, align 8
  %310 = load i8* @uc, align 1
  %311 = zext i8 %310 to i64
  %312 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %313 = atomicrmw nand i64* %312, i64 %311 monotonic
  %314 = xor i64 %313, -1
  %315 = and i64 %314, %311
  store i64 %315, i64* @sll, align 8
  %316 = load i8* @uc, align 1
  %317 = zext i8 %316 to i64
  %318 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %319 = atomicrmw nand i64* %318, i64 %317 monotonic
  %320 = xor i64 %319, -1
  %321 = and i64 %320, %317
  store i64 %321, i64* @ull, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_compare_and_swap() nounwind {
entry:
  %0 = load i8* @sc, align 1
  %1 = zext i8 %0 to i32
  %2 = load i8* @uc, align 1
  %3 = zext i8 %2 to i32
  %4 = trunc i32 %3 to i8
  %5 = trunc i32 %1 to i8
  %6 = cmpxchg i8* @sc, i8 %4, i8 %5 monotonic
  store i8 %6, i8* @sc, align 1
  %7 = load i8* @sc, align 1
  %8 = zext i8 %7 to i32
  %9 = load i8* @uc, align 1
  %10 = zext i8 %9 to i32
  %11 = trunc i32 %10 to i8
  %12 = trunc i32 %8 to i8
  %13 = cmpxchg i8* @uc, i8 %11, i8 %12 monotonic
  store i8 %13, i8* @uc, align 1
  %14 = load i8* @sc, align 1
  %15 = sext i8 %14 to i16
  %16 = zext i16 %15 to i32
  %17 = load i8* @uc, align 1
  %18 = zext i8 %17 to i32
  %19 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %20 = trunc i32 %18 to i16
  %21 = trunc i32 %16 to i16
  %22 = cmpxchg i16* %19, i16 %20, i16 %21 monotonic
  store i16 %22, i16* @ss, align 2
  %23 = load i8* @sc, align 1
  %24 = sext i8 %23 to i16
  %25 = zext i16 %24 to i32
  %26 = load i8* @uc, align 1
  %27 = zext i8 %26 to i32
  %28 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %29 = trunc i32 %27 to i16
  %30 = trunc i32 %25 to i16
  %31 = cmpxchg i16* %28, i16 %29, i16 %30 monotonic
  store i16 %31, i16* @us, align 2
  %32 = load i8* @sc, align 1
  %33 = sext i8 %32 to i32
  %34 = load i8* @uc, align 1
  %35 = zext i8 %34 to i32
  %36 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %37 = cmpxchg i32* %36, i32 %35, i32 %33 monotonic
  store i32 %37, i32* @si, align 4
  %38 = load i8* @sc, align 1
  %39 = sext i8 %38 to i32
  %40 = load i8* @uc, align 1
  %41 = zext i8 %40 to i32
  %42 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %43 = cmpxchg i32* %42, i32 %41, i32 %39 monotonic
  store i32 %43, i32* @ui, align 4
  %44 = load i8* @sc, align 1
  %45 = sext i8 %44 to i64
  %46 = load i8* @uc, align 1
  %47 = zext i8 %46 to i64
  %48 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %49 = cmpxchg i64* %48, i64 %47, i64 %45 monotonic
  store i64 %49, i64* @sl, align 8
  %50 = load i8* @sc, align 1
  %51 = sext i8 %50 to i64
  %52 = load i8* @uc, align 1
  %53 = zext i8 %52 to i64
  %54 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %55 = cmpxchg i64* %54, i64 %53, i64 %51 monotonic
  store i64 %55, i64* @ul, align 8
  %56 = load i8* @sc, align 1
  %57 = sext i8 %56 to i64
  %58 = load i8* @uc, align 1
  %59 = zext i8 %58 to i64
  %60 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %61 = cmpxchg i64* %60, i64 %59, i64 %57 monotonic
  store i64 %61, i64* @sll, align 8
  %62 = load i8* @sc, align 1
  %63 = sext i8 %62 to i64
  %64 = load i8* @uc, align 1
  %65 = zext i8 %64 to i64
  %66 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %67 = cmpxchg i64* %66, i64 %65, i64 %63 monotonic
  store i64 %67, i64* @ull, align 8
  %68 = load i8* @sc, align 1
  %69 = zext i8 %68 to i32
  %70 = load i8* @uc, align 1
  %71 = zext i8 %70 to i32
  %72 = trunc i32 %71 to i8
  %73 = trunc i32 %69 to i8
  %74 = cmpxchg i8* @sc, i8 %72, i8 %73 monotonic
  %75 = icmp eq i8 %74, %72
  %76 = zext i1 %75 to i8
  %77 = zext i8 %76 to i32
  store i32 %77, i32* @ui, align 4
  %78 = load i8* @sc, align 1
  %79 = zext i8 %78 to i32
  %80 = load i8* @uc, align 1
  %81 = zext i8 %80 to i32
  %82 = trunc i32 %81 to i8
  %83 = trunc i32 %79 to i8
  %84 = cmpxchg i8* @uc, i8 %82, i8 %83 monotonic
  %85 = icmp eq i8 %84, %82
  %86 = zext i1 %85 to i8
  %87 = zext i8 %86 to i32
  store i32 %87, i32* @ui, align 4
  %88 = load i8* @sc, align 1
  %89 = sext i8 %88 to i16
  %90 = zext i16 %89 to i32
  %91 = load i8* @uc, align 1
  %92 = zext i8 %91 to i32
  %93 = trunc i32 %92 to i8
  %94 = trunc i32 %90 to i8
  %95 = cmpxchg i8* bitcast (i16* @ss to i8*), i8 %93, i8 %94 monotonic
  %96 = icmp eq i8 %95, %93
  %97 = zext i1 %96 to i8
  %98 = zext i8 %97 to i32
  store i32 %98, i32* @ui, align 4
  %99 = load i8* @sc, align 1
  %100 = sext i8 %99 to i16
  %101 = zext i16 %100 to i32
  %102 = load i8* @uc, align 1
  %103 = zext i8 %102 to i32
  %104 = trunc i32 %103 to i8
  %105 = trunc i32 %101 to i8
  %106 = cmpxchg i8* bitcast (i16* @us to i8*), i8 %104, i8 %105 monotonic
  %107 = icmp eq i8 %106, %104
  %108 = zext i1 %107 to i8
  %109 = zext i8 %108 to i32
  store i32 %109, i32* @ui, align 4
  %110 = load i8* @sc, align 1
  %111 = sext i8 %110 to i32
  %112 = load i8* @uc, align 1
  %113 = zext i8 %112 to i32
  %114 = trunc i32 %113 to i8
  %115 = trunc i32 %111 to i8
  %116 = cmpxchg i8* bitcast (i32* @si to i8*), i8 %114, i8 %115 monotonic
  %117 = icmp eq i8 %116, %114
  %118 = zext i1 %117 to i8
  %119 = zext i8 %118 to i32
  store i32 %119, i32* @ui, align 4
  %120 = load i8* @sc, align 1
  %121 = sext i8 %120 to i32
  %122 = load i8* @uc, align 1
  %123 = zext i8 %122 to i32
  %124 = trunc i32 %123 to i8
  %125 = trunc i32 %121 to i8
  %126 = cmpxchg i8* bitcast (i32* @ui to i8*), i8 %124, i8 %125 monotonic
  %127 = icmp eq i8 %126, %124
  %128 = zext i1 %127 to i8
  %129 = zext i8 %128 to i32
  store i32 %129, i32* @ui, align 4
  %130 = load i8* @sc, align 1
  %131 = sext i8 %130 to i64
  %132 = load i8* @uc, align 1
  %133 = zext i8 %132 to i64
  %134 = trunc i64 %133 to i8
  %135 = trunc i64 %131 to i8
  %136 = cmpxchg i8* bitcast (i64* @sl to i8*), i8 %134, i8 %135 monotonic
  %137 = icmp eq i8 %136, %134
  %138 = zext i1 %137 to i8
  %139 = zext i8 %138 to i32
  store i32 %139, i32* @ui, align 4
  %140 = load i8* @sc, align 1
  %141 = sext i8 %140 to i64
  %142 = load i8* @uc, align 1
  %143 = zext i8 %142 to i64
  %144 = trunc i64 %143 to i8
  %145 = trunc i64 %141 to i8
  %146 = cmpxchg i8* bitcast (i64* @ul to i8*), i8 %144, i8 %145 monotonic
  %147 = icmp eq i8 %146, %144
  %148 = zext i1 %147 to i8
  %149 = zext i8 %148 to i32
  store i32 %149, i32* @ui, align 4
  %150 = load i8* @sc, align 1
  %151 = sext i8 %150 to i64
  %152 = load i8* @uc, align 1
  %153 = zext i8 %152 to i64
  %154 = trunc i64 %153 to i8
  %155 = trunc i64 %151 to i8
  %156 = cmpxchg i8* bitcast (i64* @sll to i8*), i8 %154, i8 %155 monotonic
  %157 = icmp eq i8 %156, %154
  %158 = zext i1 %157 to i8
  %159 = zext i8 %158 to i32
  store i32 %159, i32* @ui, align 4
  %160 = load i8* @sc, align 1
  %161 = sext i8 %160 to i64
  %162 = load i8* @uc, align 1
  %163 = zext i8 %162 to i64
  %164 = trunc i64 %163 to i8
  %165 = trunc i64 %161 to i8
  %166 = cmpxchg i8* bitcast (i64* @ull to i8*), i8 %164, i8 %165 monotonic
  %167 = icmp eq i8 %166, %164
  %168 = zext i1 %167 to i8
  %169 = zext i8 %168 to i32
  store i32 %169, i32* @ui, align 4
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
  %14 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %15 = atomicrmw xchg i64* %14, i64 1 monotonic
  store i64 %15, i64* @sll, align 8
  %16 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %17 = atomicrmw xchg i64* %16, i64 1 monotonic
  store i64 %17, i64* @ull, align 8
  fence seq_cst
  store volatile i8 0, i8* @sc, align 1
  store volatile i8 0, i8* @uc, align 1
  %18 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  store volatile i16 0, i16* %18, align 2
  %19 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  store volatile i16 0, i16* %19, align 2
  %20 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  store volatile i32 0, i32* %20, align 4
  %21 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  store volatile i32 0, i32* %21, align 4
  %22 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  store volatile i64 0, i64* %22, align 8
  %23 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  store volatile i64 0, i64* %23, align 8
  %24 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  store volatile i64 0, i64* %24, align 8
  %25 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  store volatile i64 0, i64* %25, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}
