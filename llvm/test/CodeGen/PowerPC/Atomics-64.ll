; RUN: llc < %s -verify-machineinstrs
;
; This test is disabled until PPCISelLowering learns to insert proper 64-bit
; code for ATOMIC_CMP_SWAP. Currently, it is inserting 32-bit instructions with
; 64-bit operands which causes the machine code verifier to throw a tantrum.
;
; XFAIL: *

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc64-unknown-linux-gnu"

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
  %0 = atomicrmw add ptr @sc, i8 1 monotonic
  %1 = atomicrmw add ptr @uc, i8 1 monotonic
  %2 = atomicrmw add ptr @ss, i16 1 monotonic
  %3 = atomicrmw add ptr @us, i16 1 monotonic
  %4 = atomicrmw add ptr @si, i32 1 monotonic
  %5 = atomicrmw add ptr @ui, i32 1 monotonic
  %6 = atomicrmw add ptr @sl, i64 1 monotonic
  %7 = atomicrmw add ptr @ul, i64 1 monotonic
  %8 = atomicrmw sub ptr @sc, i8 1 monotonic
  %9 = atomicrmw sub ptr @uc, i8 1 monotonic
  %10 = atomicrmw sub ptr @ss, i16 1 monotonic
  %11 = atomicrmw sub ptr @us, i16 1 monotonic
  %12 = atomicrmw sub ptr @si, i32 1 monotonic
  %13 = atomicrmw sub ptr @ui, i32 1 monotonic
  %14 = atomicrmw sub ptr @sl, i64 1 monotonic
  %15 = atomicrmw sub ptr @ul, i64 1 monotonic
  %16 = atomicrmw or ptr @sc, i8 1 monotonic
  %17 = atomicrmw or ptr @uc, i8 1 monotonic
  %18 = atomicrmw or ptr @ss, i16 1 monotonic
  %19 = atomicrmw or ptr @us, i16 1 monotonic
  %20 = atomicrmw or ptr @si, i32 1 monotonic
  %21 = atomicrmw or ptr @ui, i32 1 monotonic
  %22 = atomicrmw or ptr @sl, i64 1 monotonic
  %23 = atomicrmw or ptr @ul, i64 1 monotonic
  %24 = atomicrmw xor ptr @sc, i8 1 monotonic
  %25 = atomicrmw xor ptr @uc, i8 1 monotonic
  %26 = atomicrmw xor ptr @ss, i16 1 monotonic
  %27 = atomicrmw xor ptr @us, i16 1 monotonic
  %28 = atomicrmw xor ptr @si, i32 1 monotonic
  %29 = atomicrmw xor ptr @ui, i32 1 monotonic
  %30 = atomicrmw xor ptr @sl, i64 1 monotonic
  %31 = atomicrmw xor ptr @ul, i64 1 monotonic
  %32 = atomicrmw and ptr @sc, i8 1 monotonic
  %33 = atomicrmw and ptr @uc, i8 1 monotonic
  %34 = atomicrmw and ptr @ss, i16 1 monotonic
  %35 = atomicrmw and ptr @us, i16 1 monotonic
  %36 = atomicrmw and ptr @si, i32 1 monotonic
  %37 = atomicrmw and ptr @ui, i32 1 monotonic
  %38 = atomicrmw and ptr @sl, i64 1 monotonic
  %39 = atomicrmw and ptr @ul, i64 1 monotonic
  %40 = atomicrmw nand ptr @sc, i8 1 monotonic
  %41 = atomicrmw nand ptr @uc, i8 1 monotonic
  %42 = atomicrmw nand ptr @ss, i16 1 monotonic
  %43 = atomicrmw nand ptr @us, i16 1 monotonic
  %44 = atomicrmw nand ptr @si, i32 1 monotonic
  %45 = atomicrmw nand ptr @ui, i32 1 monotonic
  %46 = atomicrmw nand ptr @sl, i64 1 monotonic
  %47 = atomicrmw nand ptr @ul, i64 1 monotonic
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_fetch_and_op() nounwind {
entry:
  %0 = atomicrmw add ptr @sc, i8 11 monotonic
  store i8 %0, ptr @sc, align 1
  %1 = atomicrmw add ptr @uc, i8 11 monotonic
  store i8 %1, ptr @uc, align 1
  %2 = atomicrmw add ptr @ss, i16 11 monotonic
  store i16 %2, ptr @ss, align 2
  %3 = atomicrmw add ptr @us, i16 11 monotonic
  store i16 %3, ptr @us, align 2
  %4 = atomicrmw add ptr @si, i32 11 monotonic
  store i32 %4, ptr @si, align 4
  %5 = atomicrmw add ptr @ui, i32 11 monotonic
  store i32 %5, ptr @ui, align 4
  %6 = atomicrmw add ptr @sl, i64 11 monotonic
  store i64 %6, ptr @sl, align 8
  %7 = atomicrmw add ptr @ul, i64 11 monotonic
  store i64 %7, ptr @ul, align 8
  %8 = atomicrmw sub ptr @sc, i8 11 monotonic
  store i8 %8, ptr @sc, align 1
  %9 = atomicrmw sub ptr @uc, i8 11 monotonic
  store i8 %9, ptr @uc, align 1
  %10 = atomicrmw sub ptr @ss, i16 11 monotonic
  store i16 %10, ptr @ss, align 2
  %11 = atomicrmw sub ptr @us, i16 11 monotonic
  store i16 %11, ptr @us, align 2
  %12 = atomicrmw sub ptr @si, i32 11 monotonic
  store i32 %12, ptr @si, align 4
  %13 = atomicrmw sub ptr @ui, i32 11 monotonic
  store i32 %13, ptr @ui, align 4
  %14 = atomicrmw sub ptr @sl, i64 11 monotonic
  store i64 %14, ptr @sl, align 8
  %15 = atomicrmw sub ptr @ul, i64 11 monotonic
  store i64 %15, ptr @ul, align 8
  %16 = atomicrmw or ptr @sc, i8 11 monotonic
  store i8 %16, ptr @sc, align 1
  %17 = atomicrmw or ptr @uc, i8 11 monotonic
  store i8 %17, ptr @uc, align 1
  %18 = atomicrmw or ptr @ss, i16 11 monotonic
  store i16 %18, ptr @ss, align 2
  %19 = atomicrmw or ptr @us, i16 11 monotonic
  store i16 %19, ptr @us, align 2
  %20 = atomicrmw or ptr @si, i32 11 monotonic
  store i32 %20, ptr @si, align 4
  %21 = atomicrmw or ptr @ui, i32 11 monotonic
  store i32 %21, ptr @ui, align 4
  %22 = atomicrmw or ptr @sl, i64 11 monotonic
  store i64 %22, ptr @sl, align 8
  %23 = atomicrmw or ptr @ul, i64 11 monotonic
  store i64 %23, ptr @ul, align 8
  %24 = atomicrmw xor ptr @sc, i8 11 monotonic
  store i8 %24, ptr @sc, align 1
  %25 = atomicrmw xor ptr @uc, i8 11 monotonic
  store i8 %25, ptr @uc, align 1
  %26 = atomicrmw xor ptr @ss, i16 11 monotonic
  store i16 %26, ptr @ss, align 2
  %27 = atomicrmw xor ptr @us, i16 11 monotonic
  store i16 %27, ptr @us, align 2
  %28 = atomicrmw xor ptr @si, i32 11 monotonic
  store i32 %28, ptr @si, align 4
  %29 = atomicrmw xor ptr @ui, i32 11 monotonic
  store i32 %29, ptr @ui, align 4
  %30 = atomicrmw xor ptr @sl, i64 11 monotonic
  store i64 %30, ptr @sl, align 8
  %31 = atomicrmw xor ptr @ul, i64 11 monotonic
  store i64 %31, ptr @ul, align 8
  %32 = atomicrmw and ptr @sc, i8 11 monotonic
  store i8 %32, ptr @sc, align 1
  %33 = atomicrmw and ptr @uc, i8 11 monotonic
  store i8 %33, ptr @uc, align 1
  %34 = atomicrmw and ptr @ss, i16 11 monotonic
  store i16 %34, ptr @ss, align 2
  %35 = atomicrmw and ptr @us, i16 11 monotonic
  store i16 %35, ptr @us, align 2
  %36 = atomicrmw and ptr @si, i32 11 monotonic
  store i32 %36, ptr @si, align 4
  %37 = atomicrmw and ptr @ui, i32 11 monotonic
  store i32 %37, ptr @ui, align 4
  %38 = atomicrmw and ptr @sl, i64 11 monotonic
  store i64 %38, ptr @sl, align 8
  %39 = atomicrmw and ptr @ul, i64 11 monotonic
  store i64 %39, ptr @ul, align 8
  %40 = atomicrmw nand ptr @sc, i8 11 monotonic
  store i8 %40, ptr @sc, align 1
  %41 = atomicrmw nand ptr @uc, i8 11 monotonic
  store i8 %41, ptr @uc, align 1
  %42 = atomicrmw nand ptr @ss, i16 11 monotonic
  store i16 %42, ptr @ss, align 2
  %43 = atomicrmw nand ptr @us, i16 11 monotonic
  store i16 %43, ptr @us, align 2
  %44 = atomicrmw nand ptr @si, i32 11 monotonic
  store i32 %44, ptr @si, align 4
  %45 = atomicrmw nand ptr @ui, i32 11 monotonic
  store i32 %45, ptr @ui, align 4
  %46 = atomicrmw nand ptr @sl, i64 11 monotonic
  store i64 %46, ptr @sl, align 8
  %47 = atomicrmw nand ptr @ul, i64 11 monotonic
  store i64 %47, ptr @ul, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_op_and_fetch() nounwind {
entry:
  %0 = load i8, ptr @uc, align 1
  %1 = atomicrmw add ptr @sc, i8 %0 monotonic
  %2 = add i8 %1, %0
  store i8 %2, ptr @sc, align 1
  %3 = load i8, ptr @uc, align 1
  %4 = atomicrmw add ptr @uc, i8 %3 monotonic
  %5 = add i8 %4, %3
  store i8 %5, ptr @uc, align 1
  %6 = load i8, ptr @uc, align 1
  %7 = zext i8 %6 to i16
  %8 = atomicrmw add ptr @ss, i16 %7 monotonic
  %9 = add i16 %8, %7
  store i16 %9, ptr @ss, align 2
  %10 = load i8, ptr @uc, align 1
  %11 = zext i8 %10 to i16
  %12 = atomicrmw add ptr @us, i16 %11 monotonic
  %13 = add i16 %12, %11
  store i16 %13, ptr @us, align 2
  %14 = load i8, ptr @uc, align 1
  %15 = zext i8 %14 to i32
  %16 = atomicrmw add ptr @si, i32 %15 monotonic
  %17 = add i32 %16, %15
  store i32 %17, ptr @si, align 4
  %18 = load i8, ptr @uc, align 1
  %19 = zext i8 %18 to i32
  %20 = atomicrmw add ptr @ui, i32 %19 monotonic
  %21 = add i32 %20, %19
  store i32 %21, ptr @ui, align 4
  %22 = load i8, ptr @uc, align 1
  %23 = zext i8 %22 to i64
  %24 = atomicrmw add ptr @sl, i64 %23 monotonic
  %25 = add i64 %24, %23
  store i64 %25, ptr @sl, align 8
  %26 = load i8, ptr @uc, align 1
  %27 = zext i8 %26 to i64
  %28 = atomicrmw add ptr @ul, i64 %27 monotonic
  %29 = add i64 %28, %27
  store i64 %29, ptr @ul, align 8
  %30 = load i8, ptr @uc, align 1
  %31 = atomicrmw sub ptr @sc, i8 %30 monotonic
  %32 = sub i8 %31, %30
  store i8 %32, ptr @sc, align 1
  %33 = load i8, ptr @uc, align 1
  %34 = atomicrmw sub ptr @uc, i8 %33 monotonic
  %35 = sub i8 %34, %33
  store i8 %35, ptr @uc, align 1
  %36 = load i8, ptr @uc, align 1
  %37 = zext i8 %36 to i16
  %38 = atomicrmw sub ptr @ss, i16 %37 monotonic
  %39 = sub i16 %38, %37
  store i16 %39, ptr @ss, align 2
  %40 = load i8, ptr @uc, align 1
  %41 = zext i8 %40 to i16
  %42 = atomicrmw sub ptr @us, i16 %41 monotonic
  %43 = sub i16 %42, %41
  store i16 %43, ptr @us, align 2
  %44 = load i8, ptr @uc, align 1
  %45 = zext i8 %44 to i32
  %46 = atomicrmw sub ptr @si, i32 %45 monotonic
  %47 = sub i32 %46, %45
  store i32 %47, ptr @si, align 4
  %48 = load i8, ptr @uc, align 1
  %49 = zext i8 %48 to i32
  %50 = atomicrmw sub ptr @ui, i32 %49 monotonic
  %51 = sub i32 %50, %49
  store i32 %51, ptr @ui, align 4
  %52 = load i8, ptr @uc, align 1
  %53 = zext i8 %52 to i64
  %54 = atomicrmw sub ptr @sl, i64 %53 monotonic
  %55 = sub i64 %54, %53
  store i64 %55, ptr @sl, align 8
  %56 = load i8, ptr @uc, align 1
  %57 = zext i8 %56 to i64
  %58 = atomicrmw sub ptr @ul, i64 %57 monotonic
  %59 = sub i64 %58, %57
  store i64 %59, ptr @ul, align 8
  %60 = load i8, ptr @uc, align 1
  %61 = atomicrmw or ptr @sc, i8 %60 monotonic
  %62 = or i8 %61, %60
  store i8 %62, ptr @sc, align 1
  %63 = load i8, ptr @uc, align 1
  %64 = atomicrmw or ptr @uc, i8 %63 monotonic
  %65 = or i8 %64, %63
  store i8 %65, ptr @uc, align 1
  %66 = load i8, ptr @uc, align 1
  %67 = zext i8 %66 to i16
  %68 = atomicrmw or ptr @ss, i16 %67 monotonic
  %69 = or i16 %68, %67
  store i16 %69, ptr @ss, align 2
  %70 = load i8, ptr @uc, align 1
  %71 = zext i8 %70 to i16
  %72 = atomicrmw or ptr @us, i16 %71 monotonic
  %73 = or i16 %72, %71
  store i16 %73, ptr @us, align 2
  %74 = load i8, ptr @uc, align 1
  %75 = zext i8 %74 to i32
  %76 = atomicrmw or ptr @si, i32 %75 monotonic
  %77 = or i32 %76, %75
  store i32 %77, ptr @si, align 4
  %78 = load i8, ptr @uc, align 1
  %79 = zext i8 %78 to i32
  %80 = atomicrmw or ptr @ui, i32 %79 monotonic
  %81 = or i32 %80, %79
  store i32 %81, ptr @ui, align 4
  %82 = load i8, ptr @uc, align 1
  %83 = zext i8 %82 to i64
  %84 = atomicrmw or ptr @sl, i64 %83 monotonic
  %85 = or i64 %84, %83
  store i64 %85, ptr @sl, align 8
  %86 = load i8, ptr @uc, align 1
  %87 = zext i8 %86 to i64
  %88 = atomicrmw or ptr @ul, i64 %87 monotonic
  %89 = or i64 %88, %87
  store i64 %89, ptr @ul, align 8
  %90 = load i8, ptr @uc, align 1
  %91 = atomicrmw xor ptr @sc, i8 %90 monotonic
  %92 = xor i8 %91, %90
  store i8 %92, ptr @sc, align 1
  %93 = load i8, ptr @uc, align 1
  %94 = atomicrmw xor ptr @uc, i8 %93 monotonic
  %95 = xor i8 %94, %93
  store i8 %95, ptr @uc, align 1
  %96 = load i8, ptr @uc, align 1
  %97 = zext i8 %96 to i16
  %98 = atomicrmw xor ptr @ss, i16 %97 monotonic
  %99 = xor i16 %98, %97
  store i16 %99, ptr @ss, align 2
  %100 = load i8, ptr @uc, align 1
  %101 = zext i8 %100 to i16
  %102 = atomicrmw xor ptr @us, i16 %101 monotonic
  %103 = xor i16 %102, %101
  store i16 %103, ptr @us, align 2
  %104 = load i8, ptr @uc, align 1
  %105 = zext i8 %104 to i32
  %106 = atomicrmw xor ptr @si, i32 %105 monotonic
  %107 = xor i32 %106, %105
  store i32 %107, ptr @si, align 4
  %108 = load i8, ptr @uc, align 1
  %109 = zext i8 %108 to i32
  %110 = atomicrmw xor ptr @ui, i32 %109 monotonic
  %111 = xor i32 %110, %109
  store i32 %111, ptr @ui, align 4
  %112 = load i8, ptr @uc, align 1
  %113 = zext i8 %112 to i64
  %114 = atomicrmw xor ptr @sl, i64 %113 monotonic
  %115 = xor i64 %114, %113
  store i64 %115, ptr @sl, align 8
  %116 = load i8, ptr @uc, align 1
  %117 = zext i8 %116 to i64
  %118 = atomicrmw xor ptr @ul, i64 %117 monotonic
  %119 = xor i64 %118, %117
  store i64 %119, ptr @ul, align 8
  %120 = load i8, ptr @uc, align 1
  %121 = atomicrmw and ptr @sc, i8 %120 monotonic
  %122 = and i8 %121, %120
  store i8 %122, ptr @sc, align 1
  %123 = load i8, ptr @uc, align 1
  %124 = atomicrmw and ptr @uc, i8 %123 monotonic
  %125 = and i8 %124, %123
  store i8 %125, ptr @uc, align 1
  %126 = load i8, ptr @uc, align 1
  %127 = zext i8 %126 to i16
  %128 = atomicrmw and ptr @ss, i16 %127 monotonic
  %129 = and i16 %128, %127
  store i16 %129, ptr @ss, align 2
  %130 = load i8, ptr @uc, align 1
  %131 = zext i8 %130 to i16
  %132 = atomicrmw and ptr @us, i16 %131 monotonic
  %133 = and i16 %132, %131
  store i16 %133, ptr @us, align 2
  %134 = load i8, ptr @uc, align 1
  %135 = zext i8 %134 to i32
  %136 = atomicrmw and ptr @si, i32 %135 monotonic
  %137 = and i32 %136, %135
  store i32 %137, ptr @si, align 4
  %138 = load i8, ptr @uc, align 1
  %139 = zext i8 %138 to i32
  %140 = atomicrmw and ptr @ui, i32 %139 monotonic
  %141 = and i32 %140, %139
  store i32 %141, ptr @ui, align 4
  %142 = load i8, ptr @uc, align 1
  %143 = zext i8 %142 to i64
  %144 = atomicrmw and ptr @sl, i64 %143 monotonic
  %145 = and i64 %144, %143
  store i64 %145, ptr @sl, align 8
  %146 = load i8, ptr @uc, align 1
  %147 = zext i8 %146 to i64
  %148 = atomicrmw and ptr @ul, i64 %147 monotonic
  %149 = and i64 %148, %147
  store i64 %149, ptr @ul, align 8
  %150 = load i8, ptr @uc, align 1
  %151 = atomicrmw nand ptr @sc, i8 %150 monotonic
  %152 = xor i8 %151, -1
  %153 = and i8 %152, %150
  store i8 %153, ptr @sc, align 1
  %154 = load i8, ptr @uc, align 1
  %155 = atomicrmw nand ptr @uc, i8 %154 monotonic
  %156 = xor i8 %155, -1
  %157 = and i8 %156, %154
  store i8 %157, ptr @uc, align 1
  %158 = load i8, ptr @uc, align 1
  %159 = zext i8 %158 to i16
  %160 = atomicrmw nand ptr @ss, i16 %159 monotonic
  %161 = xor i16 %160, -1
  %162 = and i16 %161, %159
  store i16 %162, ptr @ss, align 2
  %163 = load i8, ptr @uc, align 1
  %164 = zext i8 %163 to i16
  %165 = atomicrmw nand ptr @us, i16 %164 monotonic
  %166 = xor i16 %165, -1
  %167 = and i16 %166, %164
  store i16 %167, ptr @us, align 2
  %168 = load i8, ptr @uc, align 1
  %169 = zext i8 %168 to i32
  %170 = atomicrmw nand ptr @si, i32 %169 monotonic
  %171 = xor i32 %170, -1
  %172 = and i32 %171, %169
  store i32 %172, ptr @si, align 4
  %173 = load i8, ptr @uc, align 1
  %174 = zext i8 %173 to i32
  %175 = atomicrmw nand ptr @ui, i32 %174 monotonic
  %176 = xor i32 %175, -1
  %177 = and i32 %176, %174
  store i32 %177, ptr @ui, align 4
  %178 = load i8, ptr @uc, align 1
  %179 = zext i8 %178 to i64
  %180 = atomicrmw nand ptr @sl, i64 %179 monotonic
  %181 = xor i64 %180, -1
  %182 = and i64 %181, %179
  store i64 %182, ptr @sl, align 8
  %183 = load i8, ptr @uc, align 1
  %184 = zext i8 %183 to i64
  %185 = atomicrmw nand ptr @ul, i64 %184 monotonic
  %186 = xor i64 %185, -1
  %187 = and i64 %186, %184
  store i64 %187, ptr @ul, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_compare_and_swap() nounwind {
entry:
  %0 = load i8, ptr @uc, align 1
  %1 = load i8, ptr @sc, align 1
  %2 = cmpxchg ptr @sc, i8 %0, i8 %1 monotonic monotonic
  store i8 %2, ptr @sc, align 1
  %3 = load i8, ptr @uc, align 1
  %4 = load i8, ptr @sc, align 1
  %5 = cmpxchg ptr @uc, i8 %3, i8 %4 monotonic monotonic
  store i8 %5, ptr @uc, align 1
  %6 = load i8, ptr @uc, align 1
  %7 = zext i8 %6 to i16
  %8 = load i8, ptr @sc, align 1
  %9 = sext i8 %8 to i16
  %10 = cmpxchg ptr @ss, i16 %7, i16 %9 monotonic monotonic
  store i16 %10, ptr @ss, align 2
  %11 = load i8, ptr @uc, align 1
  %12 = zext i8 %11 to i16
  %13 = load i8, ptr @sc, align 1
  %14 = sext i8 %13 to i16
  %15 = cmpxchg ptr @us, i16 %12, i16 %14 monotonic monotonic
  store i16 %15, ptr @us, align 2
  %16 = load i8, ptr @uc, align 1
  %17 = zext i8 %16 to i32
  %18 = load i8, ptr @sc, align 1
  %19 = sext i8 %18 to i32
  %20 = cmpxchg ptr @si, i32 %17, i32 %19 monotonic monotonic
  store i32 %20, ptr @si, align 4
  %21 = load i8, ptr @uc, align 1
  %22 = zext i8 %21 to i32
  %23 = load i8, ptr @sc, align 1
  %24 = sext i8 %23 to i32
  %25 = cmpxchg ptr @ui, i32 %22, i32 %24 monotonic monotonic
  store i32 %25, ptr @ui, align 4
  %26 = load i8, ptr @uc, align 1
  %27 = zext i8 %26 to i64
  %28 = load i8, ptr @sc, align 1
  %29 = sext i8 %28 to i64
  %30 = cmpxchg ptr @sl, i64 %27, i64 %29 monotonic monotonic
  store i64 %30, ptr @sl, align 8
  %31 = load i8, ptr @uc, align 1
  %32 = zext i8 %31 to i64
  %33 = load i8, ptr @sc, align 1
  %34 = sext i8 %33 to i64
  %35 = cmpxchg ptr @ul, i64 %32, i64 %34 monotonic monotonic
  store i64 %35, ptr @ul, align 8
  %36 = load i8, ptr @uc, align 1
  %37 = load i8, ptr @sc, align 1
  %38 = cmpxchg ptr @sc, i8 %36, i8 %37 monotonic monotonic
  %39 = icmp eq i8 %38, %36
  %40 = zext i1 %39 to i8
  %41 = zext i8 %40 to i32
  store i32 %41, ptr @ui, align 4
  %42 = load i8, ptr @uc, align 1
  %43 = load i8, ptr @sc, align 1
  %44 = cmpxchg ptr @uc, i8 %42, i8 %43 monotonic monotonic
  %45 = icmp eq i8 %44, %42
  %46 = zext i1 %45 to i8
  %47 = zext i8 %46 to i32
  store i32 %47, ptr @ui, align 4
  %48 = load i8, ptr @uc, align 1
  %49 = zext i8 %48 to i16
  %50 = load i8, ptr @sc, align 1
  %51 = sext i8 %50 to i16
  %52 = cmpxchg ptr @ss, i16 %49, i16 %51 monotonic monotonic
  %53 = icmp eq i16 %52, %49
  %54 = zext i1 %53 to i8
  %55 = zext i8 %54 to i32
  store i32 %55, ptr @ui, align 4
  %56 = load i8, ptr @uc, align 1
  %57 = zext i8 %56 to i16
  %58 = load i8, ptr @sc, align 1
  %59 = sext i8 %58 to i16
  %60 = cmpxchg ptr @us, i16 %57, i16 %59 monotonic monotonic
  %61 = icmp eq i16 %60, %57
  %62 = zext i1 %61 to i8
  %63 = zext i8 %62 to i32
  store i32 %63, ptr @ui, align 4
  %64 = load i8, ptr @uc, align 1
  %65 = zext i8 %64 to i32
  %66 = load i8, ptr @sc, align 1
  %67 = sext i8 %66 to i32
  %68 = cmpxchg ptr @si, i32 %65, i32 %67 monotonic monotonic
  %69 = icmp eq i32 %68, %65
  %70 = zext i1 %69 to i8
  %71 = zext i8 %70 to i32
  store i32 %71, ptr @ui, align 4
  %72 = load i8, ptr @uc, align 1
  %73 = zext i8 %72 to i32
  %74 = load i8, ptr @sc, align 1
  %75 = sext i8 %74 to i32
  %76 = cmpxchg ptr @ui, i32 %73, i32 %75 monotonic monotonic
  %77 = icmp eq i32 %76, %73
  %78 = zext i1 %77 to i8
  %79 = zext i8 %78 to i32
  store i32 %79, ptr @ui, align 4
  %80 = load i8, ptr @uc, align 1
  %81 = zext i8 %80 to i64
  %82 = load i8, ptr @sc, align 1
  %83 = sext i8 %82 to i64
  %84 = cmpxchg ptr @sl, i64 %81, i64 %83 monotonic monotonic
  %85 = icmp eq i64 %84, %81
  %86 = zext i1 %85 to i8
  %87 = zext i8 %86 to i32
  store i32 %87, ptr @ui, align 4
  %88 = load i8, ptr @uc, align 1
  %89 = zext i8 %88 to i64
  %90 = load i8, ptr @sc, align 1
  %91 = sext i8 %90 to i64
  %92 = cmpxchg ptr @ul, i64 %89, i64 %91 monotonic monotonic
  %93 = icmp eq i64 %92, %89
  %94 = zext i1 %93 to i8
  %95 = zext i8 %94 to i32
  store i32 %95, ptr @ui, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_lock() nounwind {
entry:
  %0 = atomicrmw xchg ptr @sc, i8 1 monotonic
  store i8 %0, ptr @sc, align 1
  %1 = atomicrmw xchg ptr @uc, i8 1 monotonic
  store i8 %1, ptr @uc, align 1
  %2 = atomicrmw xchg ptr @ss, i16 1 monotonic
  store i16 %2, ptr @ss, align 2
  %3 = atomicrmw xchg ptr @us, i16 1 monotonic
  store i16 %3, ptr @us, align 2
  %4 = atomicrmw xchg ptr @si, i32 1 monotonic
  store i32 %4, ptr @si, align 4
  %5 = atomicrmw xchg ptr @ui, i32 1 monotonic
  store i32 %5, ptr @ui, align 4
  %6 = atomicrmw xchg ptr @sl, i64 1 monotonic
  store i64 %6, ptr @sl, align 8
  %7 = atomicrmw xchg ptr @ul, i64 1 monotonic
  store i64 %7, ptr @ul, align 8
  fence seq_cst
  store volatile i8 0, ptr @sc, align 1
  store volatile i8 0, ptr @uc, align 1
  store volatile i16 0, ptr @ss, align 2
  store volatile i16 0, ptr @us, align 2
  store volatile i32 0, ptr @si, align 4
  store volatile i32 0, ptr @ui, align 4
  store volatile i64 0, ptr @sl, align 8
  store volatile i64 0, ptr @ul, align 8
  store volatile i64 0, ptr @sll, align 8
  store volatile i64 0, ptr @ull, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}
