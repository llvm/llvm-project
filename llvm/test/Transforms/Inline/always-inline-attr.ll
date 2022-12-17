; RUN: opt -passes=always-inline -S < %s | FileCheck %s


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

; After AlwaysInline the callee's attributes should be merged into caller's attibutes.

; CHECK:  define dso_local <2 x i64> @foo(ptr byval(<8 x i64>) align 64 %0) #0
; CHECK:  attributes #0 = { mustprogress uwtable "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="512"

; Function Attrs: uwtable mustprogress
define dso_local <2 x i64> @foo(ptr byval(<8 x i64>) align 64 %0) #0 {
entry:
  %kBias.addr = alloca <8 x i64>, align 64
  %indirect-arg-temp = alloca <8 x i64>, align 64
  %kBias = load <8 x i64>, ptr %0, align 64, !tbaa !2
  store <8 x i64> %kBias, ptr %kBias.addr, align 64, !tbaa !2
  %1 = load <8 x i64>, ptr %kBias.addr, align 64, !tbaa !2
  store <8 x i64> %1, ptr %indirect-arg-temp, align 64, !tbaa !2
  %call = call <2 x i64> @bar(ptr byval(<8 x i64>) align 64 %indirect-arg-temp)
  ret <2 x i64> %call
}

; Function Attrs: alwaysinline nounwind uwtable mustprogress
define internal <2 x i64> @bar(ptr byval(<8 x i64>) align 64 %0) #1 {
entry:
  %__A.addr = alloca <8 x i64>, align 64
  %__A = load <8 x i64>, ptr %0, align 64, !tbaa !2
  store <8 x i64> %__A, ptr %__A.addr, align 64, !tbaa !2
  %1 = load <8 x i64>, ptr %__A.addr, align 64, !tbaa !2
  %2 = bitcast <8 x i64> %1 to <16 x i32>
  %3 = call <16 x i8> @llvm.x86.avx512.mask.pmovs.db.512(<16 x i32> %2, <16 x i8> zeroinitializer, i16 -1)
  %4 = bitcast <16 x i8> %3 to <2 x i64>
  ret <2 x i64> %4
}

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.x86.avx512.mask.pmovs.db.512(<16 x i32>, <16 x i8>, i16) #2

attributes #0 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "prefer-vector-width"="128" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+aes,+avx,+avx2,+avx512bw,+avx512dq,+avx512f,+avx512vl,+bmi2,+cx16,+cx8,+f16c,+fma,+fxsr,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="512" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "prefer-vector-width"="128" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+aes,+avx,+avx2,+avx512f,+cx16,+cx8,+f16c,+fma,+fxsr,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }


!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
