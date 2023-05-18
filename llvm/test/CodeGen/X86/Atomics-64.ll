; RUN: llc < %s -mtriple=x86_64-apple-darwin8 > %t.x86-64
; RUN: llc < %s -mtriple=i686-apple-darwin8 -mattr=cx16 > %t.x86
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

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
  %0 = atomicrmw add ptr @sc, i8 1 monotonic
  %1 = atomicrmw add ptr @uc, i8 1 monotonic
  %2 = atomicrmw add ptr @ss, i16 1 monotonic
  %3 = atomicrmw add ptr @us, i16 1 monotonic
  %4 = atomicrmw add ptr @si, i32 1 monotonic
  %5 = atomicrmw add ptr @ui, i32 1 monotonic
  %6 = atomicrmw add ptr @sl, i64 1 monotonic
  %7 = atomicrmw add ptr @ul, i64 1 monotonic
  %8 = atomicrmw add ptr @sll, i64 1 monotonic
  %9 = atomicrmw add ptr @ull, i64 1 monotonic
  %10 = atomicrmw sub ptr @sc, i8 1 monotonic
  %11 = atomicrmw sub ptr @uc, i8 1 monotonic
  %12 = atomicrmw sub ptr @ss, i16 1 monotonic
  %13 = atomicrmw sub ptr @us, i16 1 monotonic
  %14 = atomicrmw sub ptr @si, i32 1 monotonic
  %15 = atomicrmw sub ptr @ui, i32 1 monotonic
  %16 = atomicrmw sub ptr @sl, i64 1 monotonic
  %17 = atomicrmw sub ptr @ul, i64 1 monotonic
  %18 = atomicrmw sub ptr @sll, i64 1 monotonic
  %19 = atomicrmw sub ptr @ull, i64 1 monotonic
  %20 = atomicrmw or ptr @sc, i8 1 monotonic
  %21 = atomicrmw or ptr @uc, i8 1 monotonic
  %22 = atomicrmw or ptr @ss, i16 1 monotonic
  %23 = atomicrmw or ptr @us, i16 1 monotonic
  %24 = atomicrmw or ptr @si, i32 1 monotonic
  %25 = atomicrmw or ptr @ui, i32 1 monotonic
  %26 = atomicrmw or ptr @sl, i64 1 monotonic
  %27 = atomicrmw or ptr @ul, i64 1 monotonic
  %28 = atomicrmw or ptr @sll, i64 1 monotonic
  %29 = atomicrmw or ptr @ull, i64 1 monotonic
  %30 = atomicrmw xor ptr @sc, i8 1 monotonic
  %31 = atomicrmw xor ptr @uc, i8 1 monotonic
  %32 = atomicrmw xor ptr @ss, i16 1 monotonic
  %33 = atomicrmw xor ptr @us, i16 1 monotonic
  %34 = atomicrmw xor ptr @si, i32 1 monotonic
  %35 = atomicrmw xor ptr @ui, i32 1 monotonic
  %36 = atomicrmw xor ptr @sl, i64 1 monotonic
  %37 = atomicrmw xor ptr @ul, i64 1 monotonic
  %38 = atomicrmw xor ptr @sll, i64 1 monotonic
  %39 = atomicrmw xor ptr @ull, i64 1 monotonic
  %40 = atomicrmw and ptr @sc, i8 1 monotonic
  %41 = atomicrmw and ptr @uc, i8 1 monotonic
  %42 = atomicrmw and ptr @ss, i16 1 monotonic
  %43 = atomicrmw and ptr @us, i16 1 monotonic
  %44 = atomicrmw and ptr @si, i32 1 monotonic
  %45 = atomicrmw and ptr @ui, i32 1 monotonic
  %46 = atomicrmw and ptr @sl, i64 1 monotonic
  %47 = atomicrmw and ptr @ul, i64 1 monotonic
  %48 = atomicrmw and ptr @sll, i64 1 monotonic
  %49 = atomicrmw and ptr @ull, i64 1 monotonic
  %50 = atomicrmw nand ptr @sc, i8 1 monotonic
  %51 = atomicrmw nand ptr @uc, i8 1 monotonic
  %52 = atomicrmw nand ptr @ss, i16 1 monotonic
  %53 = atomicrmw nand ptr @us, i16 1 monotonic
  %54 = atomicrmw nand ptr @si, i32 1 monotonic
  %55 = atomicrmw nand ptr @ui, i32 1 monotonic
  %56 = atomicrmw nand ptr @sl, i64 1 monotonic
  %57 = atomicrmw nand ptr @ul, i64 1 monotonic
  %58 = atomicrmw nand ptr @sll, i64 1 monotonic
  %59 = atomicrmw nand ptr @ull, i64 1 monotonic
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
  %8 = atomicrmw add ptr @sll, i64 11 monotonic
  store i64 %8, ptr @sll, align 8
  %9 = atomicrmw add ptr @ull, i64 11 monotonic
  store i64 %9, ptr @ull, align 8
  %10 = atomicrmw sub ptr @sc, i8 11 monotonic
  store i8 %10, ptr @sc, align 1
  %11 = atomicrmw sub ptr @uc, i8 11 monotonic
  store i8 %11, ptr @uc, align 1
  %12 = atomicrmw sub ptr @ss, i16 11 monotonic
  store i16 %12, ptr @ss, align 2
  %13 = atomicrmw sub ptr @us, i16 11 monotonic
  store i16 %13, ptr @us, align 2
  %14 = atomicrmw sub ptr @si, i32 11 monotonic
  store i32 %14, ptr @si, align 4
  %15 = atomicrmw sub ptr @ui, i32 11 monotonic
  store i32 %15, ptr @ui, align 4
  %16 = atomicrmw sub ptr @sl, i64 11 monotonic
  store i64 %16, ptr @sl, align 8
  %17 = atomicrmw sub ptr @ul, i64 11 monotonic
  store i64 %17, ptr @ul, align 8
  %18 = atomicrmw sub ptr @sll, i64 11 monotonic
  store i64 %18, ptr @sll, align 8
  %19 = atomicrmw sub ptr @ull, i64 11 monotonic
  store i64 %19, ptr @ull, align 8
  %20 = atomicrmw or ptr @sc, i8 11 monotonic
  store i8 %20, ptr @sc, align 1
  %21 = atomicrmw or ptr @uc, i8 11 monotonic
  store i8 %21, ptr @uc, align 1
  %22 = atomicrmw or ptr @ss, i16 11 monotonic
  store i16 %22, ptr @ss, align 2
  %23 = atomicrmw or ptr @us, i16 11 monotonic
  store i16 %23, ptr @us, align 2
  %24 = atomicrmw or ptr @si, i32 11 monotonic
  store i32 %24, ptr @si, align 4
  %25 = atomicrmw or ptr @ui, i32 11 monotonic
  store i32 %25, ptr @ui, align 4
  %26 = atomicrmw or ptr @sl, i64 11 monotonic
  store i64 %26, ptr @sl, align 8
  %27 = atomicrmw or ptr @ul, i64 11 monotonic
  store i64 %27, ptr @ul, align 8
  %28 = atomicrmw or ptr @sll, i64 11 monotonic
  store i64 %28, ptr @sll, align 8
  %29 = atomicrmw or ptr @ull, i64 11 monotonic
  store i64 %29, ptr @ull, align 8
  %30 = atomicrmw xor ptr @sc, i8 11 monotonic
  store i8 %30, ptr @sc, align 1
  %31 = atomicrmw xor ptr @uc, i8 11 monotonic
  store i8 %31, ptr @uc, align 1
  %32 = atomicrmw xor ptr @ss, i16 11 monotonic
  store i16 %32, ptr @ss, align 2
  %33 = atomicrmw xor ptr @us, i16 11 monotonic
  store i16 %33, ptr @us, align 2
  %34 = atomicrmw xor ptr @si, i32 11 monotonic
  store i32 %34, ptr @si, align 4
  %35 = atomicrmw xor ptr @ui, i32 11 monotonic
  store i32 %35, ptr @ui, align 4
  %36 = atomicrmw xor ptr @sl, i64 11 monotonic
  store i64 %36, ptr @sl, align 8
  %37 = atomicrmw xor ptr @ul, i64 11 monotonic
  store i64 %37, ptr @ul, align 8
  %38 = atomicrmw xor ptr @sll, i64 11 monotonic
  store i64 %38, ptr @sll, align 8
  %39 = atomicrmw xor ptr @ull, i64 11 monotonic
  store i64 %39, ptr @ull, align 8
  %40 = atomicrmw and ptr @sc, i8 11 monotonic
  store i8 %40, ptr @sc, align 1
  %41 = atomicrmw and ptr @uc, i8 11 monotonic
  store i8 %41, ptr @uc, align 1
  %42 = atomicrmw and ptr @ss, i16 11 monotonic
  store i16 %42, ptr @ss, align 2
  %43 = atomicrmw and ptr @us, i16 11 monotonic
  store i16 %43, ptr @us, align 2
  %44 = atomicrmw and ptr @si, i32 11 monotonic
  store i32 %44, ptr @si, align 4
  %45 = atomicrmw and ptr @ui, i32 11 monotonic
  store i32 %45, ptr @ui, align 4
  %46 = atomicrmw and ptr @sl, i64 11 monotonic
  store i64 %46, ptr @sl, align 8
  %47 = atomicrmw and ptr @ul, i64 11 monotonic
  store i64 %47, ptr @ul, align 8
  %48 = atomicrmw and ptr @sll, i64 11 monotonic
  store i64 %48, ptr @sll, align 8
  %49 = atomicrmw and ptr @ull, i64 11 monotonic
  store i64 %49, ptr @ull, align 8
  %50 = atomicrmw nand ptr @sc, i8 11 monotonic
  store i8 %50, ptr @sc, align 1
  %51 = atomicrmw nand ptr @uc, i8 11 monotonic
  store i8 %51, ptr @uc, align 1
  %52 = atomicrmw nand ptr @ss, i16 11 monotonic
  store i16 %52, ptr @ss, align 2
  %53 = atomicrmw nand ptr @us, i16 11 monotonic
  store i16 %53, ptr @us, align 2
  %54 = atomicrmw nand ptr @si, i32 11 monotonic
  store i32 %54, ptr @si, align 4
  %55 = atomicrmw nand ptr @ui, i32 11 monotonic
  store i32 %55, ptr @ui, align 4
  %56 = atomicrmw nand ptr @sl, i64 11 monotonic
  store i64 %56, ptr @sl, align 8
  %57 = atomicrmw nand ptr @ul, i64 11 monotonic
  store i64 %57, ptr @ul, align 8
  %58 = atomicrmw nand ptr @sll, i64 11 monotonic
  store i64 %58, ptr @sll, align 8
  %59 = atomicrmw nand ptr @ull, i64 11 monotonic
  store i64 %59, ptr @ull, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_op_and_fetch() nounwind {
entry:
  %0 = load i8, ptr @uc, align 1
  %1 = zext i8 %0 to i32
  %2 = trunc i32 %1 to i8
  %3 = atomicrmw add ptr @sc, i8 %2 monotonic
  %4 = add i8 %3, %2
  store i8 %4, ptr @sc, align 1
  %5 = load i8, ptr @uc, align 1
  %6 = zext i8 %5 to i32
  %7 = trunc i32 %6 to i8
  %8 = atomicrmw add ptr @uc, i8 %7 monotonic
  %9 = add i8 %8, %7
  store i8 %9, ptr @uc, align 1
  %10 = load i8, ptr @uc, align 1
  %11 = zext i8 %10 to i32
  %12 = trunc i32 %11 to i16
  %13 = atomicrmw add ptr @ss, i16 %12 monotonic
  %14 = add i16 %13, %12
  store i16 %14, ptr @ss, align 2
  %15 = load i8, ptr @uc, align 1
  %16 = zext i8 %15 to i32
  %17 = trunc i32 %16 to i16
  %18 = atomicrmw add ptr @us, i16 %17 monotonic
  %19 = add i16 %18, %17
  store i16 %19, ptr @us, align 2
  %20 = load i8, ptr @uc, align 1
  %21 = zext i8 %20 to i32
  %22 = atomicrmw add ptr @si, i32 %21 monotonic
  %23 = add i32 %22, %21
  store i32 %23, ptr @si, align 4
  %24 = load i8, ptr @uc, align 1
  %25 = zext i8 %24 to i32
  %26 = atomicrmw add ptr @ui, i32 %25 monotonic
  %27 = add i32 %26, %25
  store i32 %27, ptr @ui, align 4
  %28 = load i8, ptr @uc, align 1
  %29 = zext i8 %28 to i64
  %30 = atomicrmw add ptr @sl, i64 %29 monotonic
  %31 = add i64 %30, %29
  store i64 %31, ptr @sl, align 8
  %32 = load i8, ptr @uc, align 1
  %33 = zext i8 %32 to i64
  %34 = atomicrmw add ptr @ul, i64 %33 monotonic
  %35 = add i64 %34, %33
  store i64 %35, ptr @ul, align 8
  %36 = load i8, ptr @uc, align 1
  %37 = zext i8 %36 to i64
  %38 = atomicrmw add ptr @sll, i64 %37 monotonic
  %39 = add i64 %38, %37
  store i64 %39, ptr @sll, align 8
  %40 = load i8, ptr @uc, align 1
  %41 = zext i8 %40 to i64
  %42 = atomicrmw add ptr @ull, i64 %41 monotonic
  %43 = add i64 %42, %41
  store i64 %43, ptr @ull, align 8
  %44 = load i8, ptr @uc, align 1
  %45 = zext i8 %44 to i32
  %46 = trunc i32 %45 to i8
  %47 = atomicrmw sub ptr @sc, i8 %46 monotonic
  %48 = sub i8 %47, %46
  store i8 %48, ptr @sc, align 1
  %49 = load i8, ptr @uc, align 1
  %50 = zext i8 %49 to i32
  %51 = trunc i32 %50 to i8
  %52 = atomicrmw sub ptr @uc, i8 %51 monotonic
  %53 = sub i8 %52, %51
  store i8 %53, ptr @uc, align 1
  %54 = load i8, ptr @uc, align 1
  %55 = zext i8 %54 to i32
  %56 = trunc i32 %55 to i16
  %57 = atomicrmw sub ptr @ss, i16 %56 monotonic
  %58 = sub i16 %57, %56
  store i16 %58, ptr @ss, align 2
  %59 = load i8, ptr @uc, align 1
  %60 = zext i8 %59 to i32
  %61 = trunc i32 %60 to i16
  %62 = atomicrmw sub ptr @us, i16 %61 monotonic
  %63 = sub i16 %62, %61
  store i16 %63, ptr @us, align 2
  %64 = load i8, ptr @uc, align 1
  %65 = zext i8 %64 to i32
  %66 = atomicrmw sub ptr @si, i32 %65 monotonic
  %67 = sub i32 %66, %65
  store i32 %67, ptr @si, align 4
  %68 = load i8, ptr @uc, align 1
  %69 = zext i8 %68 to i32
  %70 = atomicrmw sub ptr @ui, i32 %69 monotonic
  %71 = sub i32 %70, %69
  store i32 %71, ptr @ui, align 4
  %72 = load i8, ptr @uc, align 1
  %73 = zext i8 %72 to i64
  %74 = atomicrmw sub ptr @sl, i64 %73 monotonic
  %75 = sub i64 %74, %73
  store i64 %75, ptr @sl, align 8
  %76 = load i8, ptr @uc, align 1
  %77 = zext i8 %76 to i64
  %78 = atomicrmw sub ptr @ul, i64 %77 monotonic
  %79 = sub i64 %78, %77
  store i64 %79, ptr @ul, align 8
  %80 = load i8, ptr @uc, align 1
  %81 = zext i8 %80 to i64
  %82 = atomicrmw sub ptr @sll, i64 %81 monotonic
  %83 = sub i64 %82, %81
  store i64 %83, ptr @sll, align 8
  %84 = load i8, ptr @uc, align 1
  %85 = zext i8 %84 to i64
  %86 = atomicrmw sub ptr @ull, i64 %85 monotonic
  %87 = sub i64 %86, %85
  store i64 %87, ptr @ull, align 8
  %88 = load i8, ptr @uc, align 1
  %89 = zext i8 %88 to i32
  %90 = trunc i32 %89 to i8
  %91 = atomicrmw or ptr @sc, i8 %90 monotonic
  %92 = or i8 %91, %90
  store i8 %92, ptr @sc, align 1
  %93 = load i8, ptr @uc, align 1
  %94 = zext i8 %93 to i32
  %95 = trunc i32 %94 to i8
  %96 = atomicrmw or ptr @uc, i8 %95 monotonic
  %97 = or i8 %96, %95
  store i8 %97, ptr @uc, align 1
  %98 = load i8, ptr @uc, align 1
  %99 = zext i8 %98 to i32
  %100 = trunc i32 %99 to i16
  %101 = atomicrmw or ptr @ss, i16 %100 monotonic
  %102 = or i16 %101, %100
  store i16 %102, ptr @ss, align 2
  %103 = load i8, ptr @uc, align 1
  %104 = zext i8 %103 to i32
  %105 = trunc i32 %104 to i16
  %106 = atomicrmw or ptr @us, i16 %105 monotonic
  %107 = or i16 %106, %105
  store i16 %107, ptr @us, align 2
  %108 = load i8, ptr @uc, align 1
  %109 = zext i8 %108 to i32
  %110 = atomicrmw or ptr @si, i32 %109 monotonic
  %111 = or i32 %110, %109
  store i32 %111, ptr @si, align 4
  %112 = load i8, ptr @uc, align 1
  %113 = zext i8 %112 to i32
  %114 = atomicrmw or ptr @ui, i32 %113 monotonic
  %115 = or i32 %114, %113
  store i32 %115, ptr @ui, align 4
  %116 = load i8, ptr @uc, align 1
  %117 = zext i8 %116 to i64
  %118 = atomicrmw or ptr @sl, i64 %117 monotonic
  %119 = or i64 %118, %117
  store i64 %119, ptr @sl, align 8
  %120 = load i8, ptr @uc, align 1
  %121 = zext i8 %120 to i64
  %122 = atomicrmw or ptr @ul, i64 %121 monotonic
  %123 = or i64 %122, %121
  store i64 %123, ptr @ul, align 8
  %124 = load i8, ptr @uc, align 1
  %125 = zext i8 %124 to i64
  %126 = atomicrmw or ptr @sll, i64 %125 monotonic
  %127 = or i64 %126, %125
  store i64 %127, ptr @sll, align 8
  %128 = load i8, ptr @uc, align 1
  %129 = zext i8 %128 to i64
  %130 = atomicrmw or ptr @ull, i64 %129 monotonic
  %131 = or i64 %130, %129
  store i64 %131, ptr @ull, align 8
  %132 = load i8, ptr @uc, align 1
  %133 = zext i8 %132 to i32
  %134 = trunc i32 %133 to i8
  %135 = atomicrmw xor ptr @sc, i8 %134 monotonic
  %136 = xor i8 %135, %134
  store i8 %136, ptr @sc, align 1
  %137 = load i8, ptr @uc, align 1
  %138 = zext i8 %137 to i32
  %139 = trunc i32 %138 to i8
  %140 = atomicrmw xor ptr @uc, i8 %139 monotonic
  %141 = xor i8 %140, %139
  store i8 %141, ptr @uc, align 1
  %142 = load i8, ptr @uc, align 1
  %143 = zext i8 %142 to i32
  %144 = trunc i32 %143 to i16
  %145 = atomicrmw xor ptr @ss, i16 %144 monotonic
  %146 = xor i16 %145, %144
  store i16 %146, ptr @ss, align 2
  %147 = load i8, ptr @uc, align 1
  %148 = zext i8 %147 to i32
  %149 = trunc i32 %148 to i16
  %150 = atomicrmw xor ptr @us, i16 %149 monotonic
  %151 = xor i16 %150, %149
  store i16 %151, ptr @us, align 2
  %152 = load i8, ptr @uc, align 1
  %153 = zext i8 %152 to i32
  %154 = atomicrmw xor ptr @si, i32 %153 monotonic
  %155 = xor i32 %154, %153
  store i32 %155, ptr @si, align 4
  %156 = load i8, ptr @uc, align 1
  %157 = zext i8 %156 to i32
  %158 = atomicrmw xor ptr @ui, i32 %157 monotonic
  %159 = xor i32 %158, %157
  store i32 %159, ptr @ui, align 4
  %160 = load i8, ptr @uc, align 1
  %161 = zext i8 %160 to i64
  %162 = atomicrmw xor ptr @sl, i64 %161 monotonic
  %163 = xor i64 %162, %161
  store i64 %163, ptr @sl, align 8
  %164 = load i8, ptr @uc, align 1
  %165 = zext i8 %164 to i64
  %166 = atomicrmw xor ptr @ul, i64 %165 monotonic
  %167 = xor i64 %166, %165
  store i64 %167, ptr @ul, align 8
  %168 = load i8, ptr @uc, align 1
  %169 = zext i8 %168 to i64
  %170 = atomicrmw xor ptr @sll, i64 %169 monotonic
  %171 = xor i64 %170, %169
  store i64 %171, ptr @sll, align 8
  %172 = load i8, ptr @uc, align 1
  %173 = zext i8 %172 to i64
  %174 = atomicrmw xor ptr @ull, i64 %173 monotonic
  %175 = xor i64 %174, %173
  store i64 %175, ptr @ull, align 8
  %176 = load i8, ptr @uc, align 1
  %177 = zext i8 %176 to i32
  %178 = trunc i32 %177 to i8
  %179 = atomicrmw and ptr @sc, i8 %178 monotonic
  %180 = and i8 %179, %178
  store i8 %180, ptr @sc, align 1
  %181 = load i8, ptr @uc, align 1
  %182 = zext i8 %181 to i32
  %183 = trunc i32 %182 to i8
  %184 = atomicrmw and ptr @uc, i8 %183 monotonic
  %185 = and i8 %184, %183
  store i8 %185, ptr @uc, align 1
  %186 = load i8, ptr @uc, align 1
  %187 = zext i8 %186 to i32
  %188 = trunc i32 %187 to i16
  %189 = atomicrmw and ptr @ss, i16 %188 monotonic
  %190 = and i16 %189, %188
  store i16 %190, ptr @ss, align 2
  %191 = load i8, ptr @uc, align 1
  %192 = zext i8 %191 to i32
  %193 = trunc i32 %192 to i16
  %194 = atomicrmw and ptr @us, i16 %193 monotonic
  %195 = and i16 %194, %193
  store i16 %195, ptr @us, align 2
  %196 = load i8, ptr @uc, align 1
  %197 = zext i8 %196 to i32
  %198 = atomicrmw and ptr @si, i32 %197 monotonic
  %199 = and i32 %198, %197
  store i32 %199, ptr @si, align 4
  %200 = load i8, ptr @uc, align 1
  %201 = zext i8 %200 to i32
  %202 = atomicrmw and ptr @ui, i32 %201 monotonic
  %203 = and i32 %202, %201
  store i32 %203, ptr @ui, align 4
  %204 = load i8, ptr @uc, align 1
  %205 = zext i8 %204 to i64
  %206 = atomicrmw and ptr @sl, i64 %205 monotonic
  %207 = and i64 %206, %205
  store i64 %207, ptr @sl, align 8
  %208 = load i8, ptr @uc, align 1
  %209 = zext i8 %208 to i64
  %210 = atomicrmw and ptr @ul, i64 %209 monotonic
  %211 = and i64 %210, %209
  store i64 %211, ptr @ul, align 8
  %212 = load i8, ptr @uc, align 1
  %213 = zext i8 %212 to i64
  %214 = atomicrmw and ptr @sll, i64 %213 monotonic
  %215 = and i64 %214, %213
  store i64 %215, ptr @sll, align 8
  %216 = load i8, ptr @uc, align 1
  %217 = zext i8 %216 to i64
  %218 = atomicrmw and ptr @ull, i64 %217 monotonic
  %219 = and i64 %218, %217
  store i64 %219, ptr @ull, align 8
  %220 = load i8, ptr @uc, align 1
  %221 = zext i8 %220 to i32
  %222 = trunc i32 %221 to i8
  %223 = atomicrmw nand ptr @sc, i8 %222 monotonic
  %224 = xor i8 %223, -1
  %225 = and i8 %224, %222
  store i8 %225, ptr @sc, align 1
  %226 = load i8, ptr @uc, align 1
  %227 = zext i8 %226 to i32
  %228 = trunc i32 %227 to i8
  %229 = atomicrmw nand ptr @uc, i8 %228 monotonic
  %230 = xor i8 %229, -1
  %231 = and i8 %230, %228
  store i8 %231, ptr @uc, align 1
  %232 = load i8, ptr @uc, align 1
  %233 = zext i8 %232 to i32
  %234 = trunc i32 %233 to i16
  %235 = atomicrmw nand ptr @ss, i16 %234 monotonic
  %236 = xor i16 %235, -1
  %237 = and i16 %236, %234
  store i16 %237, ptr @ss, align 2
  %238 = load i8, ptr @uc, align 1
  %239 = zext i8 %238 to i32
  %240 = trunc i32 %239 to i16
  %241 = atomicrmw nand ptr @us, i16 %240 monotonic
  %242 = xor i16 %241, -1
  %243 = and i16 %242, %240
  store i16 %243, ptr @us, align 2
  %244 = load i8, ptr @uc, align 1
  %245 = zext i8 %244 to i32
  %246 = atomicrmw nand ptr @si, i32 %245 monotonic
  %247 = xor i32 %246, -1
  %248 = and i32 %247, %245
  store i32 %248, ptr @si, align 4
  %249 = load i8, ptr @uc, align 1
  %250 = zext i8 %249 to i32
  %251 = atomicrmw nand ptr @ui, i32 %250 monotonic
  %252 = xor i32 %251, -1
  %253 = and i32 %252, %250
  store i32 %253, ptr @ui, align 4
  %254 = load i8, ptr @uc, align 1
  %255 = zext i8 %254 to i64
  %256 = atomicrmw nand ptr @sl, i64 %255 monotonic
  %257 = xor i64 %256, -1
  %258 = and i64 %257, %255
  store i64 %258, ptr @sl, align 8
  %259 = load i8, ptr @uc, align 1
  %260 = zext i8 %259 to i64
  %261 = atomicrmw nand ptr @ul, i64 %260 monotonic
  %262 = xor i64 %261, -1
  %263 = and i64 %262, %260
  store i64 %263, ptr @ul, align 8
  %264 = load i8, ptr @uc, align 1
  %265 = zext i8 %264 to i64
  %266 = atomicrmw nand ptr @sll, i64 %265 monotonic
  %267 = xor i64 %266, -1
  %268 = and i64 %267, %265
  store i64 %268, ptr @sll, align 8
  %269 = load i8, ptr @uc, align 1
  %270 = zext i8 %269 to i64
  %271 = atomicrmw nand ptr @ull, i64 %270 monotonic
  %272 = xor i64 %271, -1
  %273 = and i64 %272, %270
  store i64 %273, ptr @ull, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_compare_and_swap() nounwind {
entry:
  %0 = load i8, ptr @sc, align 1
  %1 = zext i8 %0 to i32
  %2 = load i8, ptr @uc, align 1
  %3 = zext i8 %2 to i32
  %4 = trunc i32 %3 to i8
  %5 = trunc i32 %1 to i8
  %pair6 = cmpxchg ptr @sc, i8 %4, i8 %5 monotonic monotonic
  %6 = extractvalue { i8, i1 } %pair6, 0
  store i8 %6, ptr @sc, align 1
  %7 = load i8, ptr @sc, align 1
  %8 = zext i8 %7 to i32
  %9 = load i8, ptr @uc, align 1
  %10 = zext i8 %9 to i32
  %11 = trunc i32 %10 to i8
  %12 = trunc i32 %8 to i8
  %pair13 = cmpxchg ptr @uc, i8 %11, i8 %12 monotonic monotonic
  %13 = extractvalue { i8, i1 } %pair13, 0
  store i8 %13, ptr @uc, align 1
  %14 = load i8, ptr @sc, align 1
  %15 = sext i8 %14 to i16
  %16 = zext i16 %15 to i32
  %17 = load i8, ptr @uc, align 1
  %18 = zext i8 %17 to i32
  %19 = trunc i32 %18 to i16
  %20 = trunc i32 %16 to i16
  %pair22 = cmpxchg ptr @ss, i16 %19, i16 %20 monotonic monotonic
  %21 = extractvalue { i16, i1 } %pair22, 0
  store i16 %21, ptr @ss, align 2
  %22 = load i8, ptr @sc, align 1
  %23 = sext i8 %22 to i16
  %24 = zext i16 %23 to i32
  %25 = load i8, ptr @uc, align 1
  %26 = zext i8 %25 to i32
  %27 = trunc i32 %26 to i16
  %28 = trunc i32 %24 to i16
  %pair31 = cmpxchg ptr @us, i16 %27, i16 %28 monotonic monotonic
  %29 = extractvalue { i16, i1 } %pair31, 0
  store i16 %29, ptr @us, align 2
  %30 = load i8, ptr @sc, align 1
  %31 = sext i8 %30 to i32
  %32 = load i8, ptr @uc, align 1
  %33 = zext i8 %32 to i32
  %pair37 = cmpxchg ptr @si, i32 %33, i32 %31 monotonic monotonic
  %34 = extractvalue { i32, i1 } %pair37, 0
  store i32 %34, ptr @si, align 4
  %35 = load i8, ptr @sc, align 1
  %36 = sext i8 %35 to i32
  %37 = load i8, ptr @uc, align 1
  %38 = zext i8 %37 to i32
  %pair43 = cmpxchg ptr @ui, i32 %38, i32 %36 monotonic monotonic
  %39 = extractvalue { i32, i1 } %pair43, 0
  store i32 %39, ptr @ui, align 4
  %40 = load i8, ptr @sc, align 1
  %41 = sext i8 %40 to i64
  %42 = load i8, ptr @uc, align 1
  %43 = zext i8 %42 to i64
  %pair49 = cmpxchg ptr @sl, i64 %43, i64 %41 monotonic monotonic
  %44 = extractvalue { i64, i1 } %pair49, 0
  store i64 %44, ptr @sl, align 8
  %45 = load i8, ptr @sc, align 1
  %46 = sext i8 %45 to i64
  %47 = load i8, ptr @uc, align 1
  %48 = zext i8 %47 to i64
  %pair55 = cmpxchg ptr @ul, i64 %48, i64 %46 monotonic monotonic
  %49 = extractvalue { i64, i1 } %pair55, 0
  store i64 %49, ptr @ul, align 8
  %50 = load i8, ptr @sc, align 1
  %51 = sext i8 %50 to i64
  %52 = load i8, ptr @uc, align 1
  %53 = zext i8 %52 to i64
  %pair61 = cmpxchg ptr @sll, i64 %53, i64 %51 monotonic monotonic
  %54 = extractvalue { i64, i1 } %pair61, 0
  store i64 %54, ptr @sll, align 8
  %55 = load i8, ptr @sc, align 1
  %56 = sext i8 %55 to i64
  %57 = load i8, ptr @uc, align 1
  %58 = zext i8 %57 to i64
  %pair67 = cmpxchg ptr @ull, i64 %58, i64 %56 monotonic monotonic
  %59 = extractvalue { i64, i1 } %pair67, 0
  store i64 %59, ptr @ull, align 8
  %60 = load i8, ptr @sc, align 1
  %61 = zext i8 %60 to i32
  %62 = load i8, ptr @uc, align 1
  %63 = zext i8 %62 to i32
  %64 = trunc i32 %63 to i8
  %65 = trunc i32 %61 to i8
  %pair74 = cmpxchg ptr @sc, i8 %64, i8 %65 monotonic monotonic
  %66 = extractvalue { i8, i1 } %pair74, 0
  %67 = icmp eq i8 %66, %64
  %68 = zext i1 %67 to i8
  %69 = zext i8 %68 to i32
  store i32 %69, ptr @ui, align 4
  %70 = load i8, ptr @sc, align 1
  %71 = zext i8 %70 to i32
  %72 = load i8, ptr @uc, align 1
  %73 = zext i8 %72 to i32
  %74 = trunc i32 %73 to i8
  %75 = trunc i32 %71 to i8
  %pair84 = cmpxchg ptr @uc, i8 %74, i8 %75 monotonic monotonic
  %76 = extractvalue { i8, i1 } %pair84, 0
  %77 = icmp eq i8 %76, %74
  %78 = zext i1 %77 to i8
  %79 = zext i8 %78 to i32
  store i32 %79, ptr @ui, align 4
  %80 = load i8, ptr @sc, align 1
  %81 = sext i8 %80 to i16
  %82 = zext i16 %81 to i32
  %83 = load i8, ptr @uc, align 1
  %84 = zext i8 %83 to i32
  %85 = trunc i32 %84 to i8
  %86 = trunc i32 %82 to i8
  %pair95 = cmpxchg ptr @ss, i8 %85, i8 %86 monotonic monotonic
  %87 = extractvalue { i8, i1 } %pair95, 0
  %88 = icmp eq i8 %87, %85
  %89 = zext i1 %88 to i8
  %90 = zext i8 %89 to i32
  store i32 %90, ptr @ui, align 4
  %91 = load i8, ptr @sc, align 1
  %92 = sext i8 %91 to i16
  %93 = zext i16 %92 to i32
  %94 = load i8, ptr @uc, align 1
  %95 = zext i8 %94 to i32
  %96 = trunc i32 %95 to i8
  %97 = trunc i32 %93 to i8
  %pair106 = cmpxchg ptr @us, i8 %96, i8 %97 monotonic monotonic
  %98 = extractvalue { i8, i1 } %pair106, 0
  %99 = icmp eq i8 %98, %96
  %100 = zext i1 %99 to i8
  %101 = zext i8 %100 to i32
  store i32 %101, ptr @ui, align 4
  %102 = load i8, ptr @sc, align 1
  %103 = sext i8 %102 to i32
  %104 = load i8, ptr @uc, align 1
  %105 = zext i8 %104 to i32
  %106 = trunc i32 %105 to i8
  %107 = trunc i32 %103 to i8
  %pair116 = cmpxchg ptr @si, i8 %106, i8 %107 monotonic monotonic
  %108 = extractvalue { i8, i1 } %pair116, 0
  %109 = icmp eq i8 %108, %106
  %110 = zext i1 %109 to i8
  %111 = zext i8 %110 to i32
  store i32 %111, ptr @ui, align 4
  %112 = load i8, ptr @sc, align 1
  %113 = sext i8 %112 to i32
  %114 = load i8, ptr @uc, align 1
  %115 = zext i8 %114 to i32
  %116 = trunc i32 %115 to i8
  %117 = trunc i32 %113 to i8
  %pair126 = cmpxchg ptr @ui, i8 %116, i8 %117 monotonic monotonic
  %118 = extractvalue { i8, i1 } %pair126, 0
  %119 = icmp eq i8 %118, %116
  %120 = zext i1 %119 to i8
  %121 = zext i8 %120 to i32
  store i32 %121, ptr @ui, align 4
  %122 = load i8, ptr @sc, align 1
  %123 = sext i8 %122 to i64
  %124 = load i8, ptr @uc, align 1
  %125 = zext i8 %124 to i64
  %126 = trunc i64 %125 to i8
  %127 = trunc i64 %123 to i8
  %pair136 = cmpxchg ptr @sl, i8 %126, i8 %127 monotonic monotonic
  %128 = extractvalue { i8, i1 } %pair136, 0
  %129 = icmp eq i8 %128, %126
  %130 = zext i1 %129 to i8
  %131 = zext i8 %130 to i32
  store i32 %131, ptr @ui, align 4
  %132 = load i8, ptr @sc, align 1
  %133 = sext i8 %132 to i64
  %134 = load i8, ptr @uc, align 1
  %135 = zext i8 %134 to i64
  %136 = trunc i64 %135 to i8
  %137 = trunc i64 %133 to i8
  %pair146 = cmpxchg ptr @ul, i8 %136, i8 %137 monotonic monotonic
  %138 = extractvalue { i8, i1 } %pair146, 0
  %139 = icmp eq i8 %138, %136
  %140 = zext i1 %139 to i8
  %141 = zext i8 %140 to i32
  store i32 %141, ptr @ui, align 4
  %142 = load i8, ptr @sc, align 1
  %143 = sext i8 %142 to i64
  %144 = load i8, ptr @uc, align 1
  %145 = zext i8 %144 to i64
  %146 = trunc i64 %145 to i8
  %147 = trunc i64 %143 to i8
  %pair156 = cmpxchg ptr @sll, i8 %146, i8 %147 monotonic monotonic
  %148 = extractvalue { i8, i1 } %pair156, 0
  %149 = icmp eq i8 %148, %146
  %150 = zext i1 %149 to i8
  %151 = zext i8 %150 to i32
  store i32 %151, ptr @ui, align 4
  %152 = load i8, ptr @sc, align 1
  %153 = sext i8 %152 to i64
  %154 = load i8, ptr @uc, align 1
  %155 = zext i8 %154 to i64
  %156 = trunc i64 %155 to i8
  %157 = trunc i64 %153 to i8
  %pair166 = cmpxchg ptr @ull, i8 %156, i8 %157 monotonic monotonic
  %158 = extractvalue { i8, i1 } %pair166, 0
  %159 = icmp eq i8 %158, %156
  %160 = zext i1 %159 to i8
  %161 = zext i8 %160 to i32
  store i32 %161, ptr @ui, align 4
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
  %8 = atomicrmw xchg ptr @sll, i64 1 monotonic
  store i64 %8, ptr @sll, align 8
  %9 = atomicrmw xchg ptr @ull, i64 1 monotonic
  store i64 %9, ptr @ull, align 8
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
