; RUN: llc < %s -mtriple=x86_64-apple-darwin8 -misched-print-dags -o - 2>&1 > /dev/null | FileCheck %s

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

; Regression Test for PR92368.
;
; CHECK: SU(75):   CMP8rr %49:gr8, %48:gr8, implicit-def $eflags
; CHECK:   Predecessors:
; CHECK-NEXT:    SU(73): Data Latency=0 Reg=%49
; CHECK-NEXT:    SU(74): Out  Latency=0
; CHECK-NEXT:    SU(72): Out  Latency=0
; CHECK-NEXT:    SU(70): Data Latency=4 Reg=%48
define void @misched_bug() nounwind {
entry:
  %0 = load i8, i8* @sc, align 1
  %1 = zext i8 %0 to i32
  %2 = load i8, i8* @uc, align 1
  %3 = zext i8 %2 to i32
  %4 = trunc i32 %3 to i8
  %5 = trunc i32 %1 to i8
  %pair6 = cmpxchg i8* @sc, i8 %4, i8 %5 monotonic monotonic
  %6 = extractvalue { i8, i1 } %pair6, 0
  store i8 %6, i8* @sc, align 1
  %7 = load i8, i8* @sc, align 1
  %8 = zext i8 %7 to i32
  %9 = load i8, i8* @uc, align 1
  %10 = zext i8 %9 to i32
  %11 = trunc i32 %10 to i8
  %12 = trunc i32 %8 to i8
  %pair13 = cmpxchg i8* @uc, i8 %11, i8 %12 monotonic monotonic
  %13 = extractvalue { i8, i1 } %pair13, 0
  store i8 %13, i8* @uc, align 1
  %14 = load i8, i8* @sc, align 1
  %15 = sext i8 %14 to i16
  %16 = zext i16 %15 to i32
  %17 = load i8, i8* @uc, align 1
  %18 = zext i8 %17 to i32
  %19 = bitcast i8* bitcast (i16* @ss to i8*) to i16*
  %20 = trunc i32 %18 to i16
  %21 = trunc i32 %16 to i16
  %pair22 = cmpxchg i16* %19, i16 %20, i16 %21 monotonic monotonic
  %22 = extractvalue { i16, i1 } %pair22, 0
  store i16 %22, i16* @ss, align 2
  %23 = load i8, i8* @sc, align 1
  %24 = sext i8 %23 to i16
  %25 = zext i16 %24 to i32
  %26 = load i8, i8* @uc, align 1
  %27 = zext i8 %26 to i32
  %28 = bitcast i8* bitcast (i16* @us to i8*) to i16*
  %29 = trunc i32 %27 to i16
  %30 = trunc i32 %25 to i16
  %pair31 = cmpxchg i16* %28, i16 %29, i16 %30 monotonic monotonic
  %31 = extractvalue { i16, i1 } %pair31, 0
  store i16 %31, i16* @us, align 2
  %32 = load i8, i8* @sc, align 1
  %33 = sext i8 %32 to i32
  %34 = load i8, i8* @uc, align 1
  %35 = zext i8 %34 to i32
  %36 = bitcast i8* bitcast (i32* @si to i8*) to i32*
  %pair37 = cmpxchg i32* %36, i32 %35, i32 %33 monotonic monotonic
  %37 = extractvalue { i32, i1 } %pair37, 0
  store i32 %37, i32* @si, align 4
  %38 = load i8, i8* @sc, align 1
  %39 = sext i8 %38 to i32
  %40 = load i8, i8* @uc, align 1
  %41 = zext i8 %40 to i32
  %42 = bitcast i8* bitcast (i32* @ui to i8*) to i32*
  %pair43 = cmpxchg i32* %42, i32 %41, i32 %39 monotonic monotonic
  %43 = extractvalue { i32, i1 } %pair43, 0
  store i32 %43, i32* @ui, align 4
  %44 = load i8, i8* @sc, align 1
  %45 = sext i8 %44 to i64
  %46 = load i8, i8* @uc, align 1
  %47 = zext i8 %46 to i64
  %48 = bitcast i8* bitcast (i64* @sl to i8*) to i64*
  %pair49 = cmpxchg i64* %48, i64 %47, i64 %45 monotonic monotonic
  %49 = extractvalue { i64, i1 } %pair49, 0
  store i64 %49, i64* @sl, align 8
  %50 = load i8, i8* @sc, align 1
  %51 = sext i8 %50 to i64
  %52 = load i8, i8* @uc, align 1
  %53 = zext i8 %52 to i64
  %54 = bitcast i8* bitcast (i64* @ul to i8*) to i64*
  %pair55 = cmpxchg i64* %54, i64 %53, i64 %51 monotonic monotonic
  %55 = extractvalue { i64, i1 } %pair55, 0
  store i64 %55, i64* @ul, align 8
  %56 = load i8, i8* @sc, align 1
  %57 = sext i8 %56 to i64
  %58 = load i8, i8* @uc, align 1
  %59 = zext i8 %58 to i64
  %60 = bitcast i8* bitcast (i64* @sll to i8*) to i64*
  %pair61 = cmpxchg i64* %60, i64 %59, i64 %57 monotonic monotonic
  %61 = extractvalue { i64, i1 } %pair61, 0
  store i64 %61, i64* @sll, align 8
  %62 = load i8, i8* @sc, align 1
  %63 = sext i8 %62 to i64
  %64 = load i8, i8* @uc, align 1
  %65 = zext i8 %64 to i64
  %66 = bitcast i8* bitcast (i64* @ull to i8*) to i64*
  %pair67 = cmpxchg i64* %66, i64 %65, i64 %63 monotonic monotonic
  %67 = extractvalue { i64, i1 } %pair67, 0
  store i64 %67, i64* @ull, align 8
  %68 = load i8, i8* @sc, align 1
  %69 = zext i8 %68 to i32
  %70 = load i8, i8* @uc, align 1
  %71 = zext i8 %70 to i32
  %72 = trunc i32 %71 to i8
  %73 = trunc i32 %69 to i8
  %pair74 = cmpxchg i8* @sc, i8 %72, i8 %73 monotonic monotonic
  %74 = extractvalue { i8, i1 } %pair74, 0
  %75 = icmp eq i8 %74, %72
  %76 = zext i1 %75 to i8
  %77 = zext i8 %76 to i32
  store i32 %77, i32* @ui, align 4
  %78 = load i8, i8* @sc, align 1
  %79 = zext i8 %78 to i32
  %80 = load i8, i8* @uc, align 1
  %81 = zext i8 %80 to i32
  %82 = trunc i32 %81 to i8
  %83 = trunc i32 %79 to i8
  %pair84 = cmpxchg i8* @uc, i8 %82, i8 %83 monotonic monotonic
  %84 = extractvalue { i8, i1 } %pair84, 0
  %85 = icmp eq i8 %84, %82
  %86 = zext i1 %85 to i8
  %87 = zext i8 %86 to i32
  store i32 %87, i32* @ui, align 4
  %88 = load i8, i8* @sc, align 1
  %89 = sext i8 %88 to i16
  %90 = zext i16 %89 to i32
  %91 = load i8, i8* @uc, align 1
  %92 = zext i8 %91 to i32
  %93 = trunc i32 %92 to i8
  %94 = trunc i32 %90 to i8
  %pair95 = cmpxchg i8* bitcast (i16* @ss to i8*), i8 %93, i8 %94 monotonic monotonic
  %95 = extractvalue { i8, i1 } %pair95, 0
  %96 = icmp eq i8 %95, %93
  %97 = zext i1 %96 to i8
  %98 = zext i8 %97 to i32
  store i32 %98, i32* @ui, align 4
  %99 = load i8, i8* @sc, align 1
  %100 = sext i8 %99 to i16
  %101 = zext i16 %100 to i32
  %102 = load i8, i8* @uc, align 1
  %103 = zext i8 %102 to i32
  %104 = trunc i32 %103 to i8
  %105 = trunc i32 %101 to i8
  %pair106 = cmpxchg i8* bitcast (i16* @us to i8*), i8 %104, i8 %105 monotonic monotonic
  %106 = extractvalue { i8, i1 } %pair106, 0
  %107 = icmp eq i8 %106, %104
  %108 = zext i1 %107 to i8
  %109 = zext i8 %108 to i32
  store i32 %109, i32* @ui, align 4
  %110 = load i8, i8* @sc, align 1
  %111 = sext i8 %110 to i32
  %112 = load i8, i8* @uc, align 1
  %113 = zext i8 %112 to i32
  %114 = trunc i32 %113 to i8
  %115 = trunc i32 %111 to i8
  %pair116 = cmpxchg i8* bitcast (i32* @si to i8*), i8 %114, i8 %115 monotonic monotonic
  %116 = extractvalue { i8, i1 } %pair116, 0
  %117 = icmp eq i8 %116, %114
  %118 = zext i1 %117 to i8
  %119 = zext i8 %118 to i32
  store i32 %119, i32* @ui, align 4
  %120 = load i8, i8* @sc, align 1
  %121 = sext i8 %120 to i32
  %122 = load i8, i8* @uc, align 1
  %123 = zext i8 %122 to i32
  %124 = trunc i32 %123 to i8
  %125 = trunc i32 %121 to i8
  %pair126 = cmpxchg i8* bitcast (i32* @ui to i8*), i8 %124, i8 %125 monotonic monotonic
  %126 = extractvalue { i8, i1 } %pair126, 0
  %127 = icmp eq i8 %126, %124
  %128 = zext i1 %127 to i8
  %129 = zext i8 %128 to i32
  store i32 %129, i32* @ui, align 4
  %130 = load i8, i8* @sc, align 1
  %131 = sext i8 %130 to i64
  %132 = load i8, i8* @uc, align 1
  %133 = zext i8 %132 to i64
  %134 = trunc i64 %133 to i8
  %135 = trunc i64 %131 to i8
  %pair136 = cmpxchg i8* bitcast (i64* @sl to i8*), i8 %134, i8 %135 monotonic monotonic
  %136 = extractvalue { i8, i1 } %pair136, 0
  %137 = icmp eq i8 %136, %134
  %138 = zext i1 %137 to i8
  %139 = zext i8 %138 to i32
  store i32 %139, i32* @ui, align 4
  %140 = load i8, i8* @sc, align 1
  %141 = sext i8 %140 to i64
  %142 = load i8, i8* @uc, align 1
  %143 = zext i8 %142 to i64
  %144 = trunc i64 %143 to i8
  %145 = trunc i64 %141 to i8
  %pair146 = cmpxchg i8* bitcast (i64* @ul to i8*), i8 %144, i8 %145 monotonic monotonic
  %146 = extractvalue { i8, i1 } %pair146, 0
  %147 = icmp eq i8 %146, %144
  %148 = zext i1 %147 to i8
  %149 = zext i8 %148 to i32
  store i32 %149, i32* @ui, align 4
  %150 = load i8, i8* @sc, align 1
  %151 = sext i8 %150 to i64
  %152 = load i8, i8* @uc, align 1
  %153 = zext i8 %152 to i64
  %154 = trunc i64 %153 to i8
  %155 = trunc i64 %151 to i8
  %pair156 = cmpxchg i8* bitcast (i64* @sll to i8*), i8 %154, i8 %155 monotonic monotonic
  %156 = extractvalue { i8, i1 } %pair156, 0
  %157 = icmp eq i8 %156, %154
  %158 = zext i1 %157 to i8
  %159 = zext i8 %158 to i32
  store i32 %159, i32* @ui, align 4
  %160 = load i8, i8* @sc, align 1
  %161 = sext i8 %160 to i64
  %162 = load i8, i8* @uc, align 1
  %163 = zext i8 %162 to i64
  %164 = trunc i64 %163 to i8
  %165 = trunc i64 %161 to i8
  %pair166 = cmpxchg i8* bitcast (i64* @ull to i8*), i8 %164, i8 %165 monotonic monotonic
  %166 = extractvalue { i8, i1 } %pair166, 0
  %167 = icmp eq i8 %166, %164
  %168 = zext i1 %167 to i8
  %169 = zext i8 %168 to i32
  store i32 %169, i32* @ui, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}
