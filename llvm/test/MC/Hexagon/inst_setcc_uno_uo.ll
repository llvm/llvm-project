;; RUN: llc -mv79 -mhvx %s -o - | FileCheck %s
source_filename = "isnan.c"
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local void @foo(ptr noundef readonly captures(none) %a, ptr noundef writeonly captures(none) %isnan_a) local_unnamed_addr #0 {
entry:
  %arrayidx = getelementptr inbounds nuw float, ptr %a, i32 0
  %0 = load <32 x float>, ptr %arrayidx, align 4
  %.ripple.vectorized = fcmp uno <32 x float> %0, zeroinitializer
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %isnan_a, i32 0
  %storedv.ripple.LS.instance = zext <32 x i1> %.ripple.vectorized to <32 x i8>
  store <32 x i8> %storedv.ripple.LS.instance, ptr %arrayidx1, align 1
  ret void
}

;; CHECK: vcmp.eq

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hmx,+hvx-length128b,+hvxv79,+v79,-long-calls" }
