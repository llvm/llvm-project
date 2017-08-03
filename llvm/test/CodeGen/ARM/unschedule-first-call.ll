; RUN: llc -verify-machineinstrs < %s
; PR30911

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv6kz--linux-gnueabihf"

; Function Attrs: nounwind
define void @dradbg(i32, i32, float*, float*, float*, float*, float*) #0 {
  br i1 undef, label %.critedge, label %8

.critedge:                                        ; preds = %7
  %.mux2 = select i1 undef, i1 undef, i1 true
  br label %8

; <label>:8:                                      ; preds = %.critedge, %7
  %9 = getelementptr float, float* %3, i64 undef
  %10 = ptrtoint float* %9 to i32
  %11 = icmp ule i32 %10, undef
  %12 = getelementptr float, float* %5, i64 undef
  %13 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 undef, i64 undef)
  %14 = extractvalue { i64, i1 } %13, 0
  %15 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %14, i64 1)
  %16 = extractvalue { i64, i1 } %15, 0
  %17 = icmp slt i64 1, %16
  %18 = select i1 %17, i64 1, i64 %16
  %19 = sext i32 %1 to i64
  %20 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %18, i64 %19)
  %21 = extractvalue { i64, i1 } %20, 0
  %22 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %21, i64 0)
  %23 = extractvalue { i64, i1 } %22, 0
  %24 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %23, i64 undef)
  %25 = extractvalue { i64, i1 } %24, 0
  %26 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %25, i64 0)
  %27 = extractvalue { i64, i1 } %26, 0
  %28 = getelementptr float, float* %3, i64 %27
  %29 = ptrtoint float* %12 to i32
  %30 = ptrtoint float* %28 to i32
  %31 = icmp ule i32 %29, %30
  %32 = or i1 %11, %31
  %33 = and i1 false, %32
  %34 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 0, i64 undef)
  %35 = extractvalue { i64, i1 } %34, 0
  %36 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %35, i64 1)
  %37 = extractvalue { i64, i1 } %36, 0
  %38 = icmp slt i64 1, %37
  %39 = select i1 %38, i64 1, i64 %37
  %40 = sext i32 %1 to i64
  %41 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %39, i64 %40)
  %42 = extractvalue { i64, i1 } %41, 0
  %43 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %42, i64 0)
  %44 = extractvalue { i64, i1 } %43, 0
  %45 = sext i32 %0 to i64
  %46 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %44, i64 %45)
  %47 = extractvalue { i64, i1 } %46, 0
  %48 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %47, i64 0)
  %49 = extractvalue { i64, i1 } %48, 0
  %50 = getelementptr float, float* %5, i64 %49
  %51 = ptrtoint float* %50 to i32
  %52 = icmp ule i32 undef, %51
  %53 = getelementptr float, float* %4, i64 undef
  %54 = ptrtoint float* %53 to i32
  %55 = icmp ule i32 undef, %54
  %56 = or i1 %52, %55
  %57 = and i1 %33, %56
  %58 = getelementptr float, float* %2, i64 undef
  %59 = ptrtoint float* %58 to i32
  %60 = icmp ule i32 %59, undef
  %61 = select i1 undef, i64 undef, i64 0
  %62 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %61, i64 undef)
  %63 = extractvalue { i64, i1 } %62, 0
  %64 = call { i64, i1 } @llvm.ssub.with.overflow.i64(i64 undef, i64 1)
  %65 = extractvalue { i64, i1 } %64, 0
  %66 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %63, i64 %65)
  %67 = extractvalue { i64, i1 } %66, 0
  %68 = sext i32 %0 to i64
  %69 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %67, i64 %68)
  %70 = extractvalue { i64, i1 } %69, 0
  %71 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %70, i64 0)
  %72 = extractvalue { i64, i1 } %71, 0
  %73 = getelementptr float, float* %5, i64 %72
  %74 = ptrtoint float* %73 to i32
  %75 = icmp ule i32 %74, undef
  %76 = or i1 %60, %75
  %77 = and i1 %57, %76
  %78 = getelementptr float, float* %6, i64 undef
  %79 = ptrtoint float* %78 to i32
  %80 = icmp ule i32 %79, undef
  %81 = getelementptr float, float* %5, i64 undef
  %82 = ptrtoint float* %81 to i32
  %83 = icmp ule i32 %82, undef
  %84 = or i1 %80, %83
  %85 = and i1 %77, %84
  %86 = and i1 %85, undef
  %87 = and i1 %86, undef
  %88 = and i1 %87, undef
  %89 = and i1 %88, undef
  %90 = and i1 %89, undef
  %91 = and i1 %90, undef
  %92 = and i1 %91, undef
  %93 = and i1 %92, undef
  %94 = and i1 %93, undef
  %95 = and i1 %94, undef
  br i1 %95, label %97, label %96

; <label>:96:                                     ; preds = %8
  br i1 undef, label %.critedge122, label %.critedge110

.critedge122:                                     ; preds = %.critedge122, %96
  br i1 false, label %.critedge122, label %.critedge110

.critedge110:                                     ; preds = %.critedge219, %97, %.critedge122, %96
  ret void

; <label>:97:                                     ; preds = %8
  br i1 undef, label %.critedge219, label %.critedge110

.critedge219:                                     ; preds = %.critedge219, %97
  %.pr287 = phi i1 [ undef, %.critedge219 ], [ true, %97 ]
  br i1 %.pr287, label %.critedge219, label %.critedge110
}

; Function Attrs: nounwind readnone
declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64) #1

; Function Attrs: nounwind readnone
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64) #1

; Function Attrs: nounwind readnone
declare { i64, i1 } @llvm.ssub.with.overflow.i64(i64, i64) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "polly-optimized" "stack-protector-buffer-size"="8" "target-cpu"="arm1176jzf-s" "target-features"="+dsp,+strict-align,+vfp2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (trunk 285923) (llvm/trunk 285921)"}
