; RUN: llc -O2 -march=hexagon < %s | FileCheck %s
; Make sure we don't use setbit to add offsets to stack objects.
; CHECK-NOT: setbit

target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = alloca [64 x float], align 16
  %v1 = bitcast [64 x float]* %v0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 256, i8* %v1) #1
  %v2 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 8
  store float 0.000000e+00, float* %v2, align 16, !tbaa !0
  %v3 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 0
  store float 0.000000e+00, float* %v3, align 16, !tbaa !0
  %v4 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 9
  store float 0.000000e+00, float* %v4, align 4, !tbaa !0
  %v5 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 1
  store float 0.000000e+00, float* %v5, align 4, !tbaa !0
  %v6 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 10
  store float 0.000000e+00, float* %v6, align 8, !tbaa !0
  %v7 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 2
  store float 0.000000e+00, float* %v7, align 8, !tbaa !0
  %v8 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 11
  store float 1.000000e+00, float* %v8, align 4, !tbaa !0
  %v9 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 3
  store float 1.000000e+00, float* %v9, align 4, !tbaa !0
  %v10 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 12
  store float 0.000000e+00, float* %v10, align 16, !tbaa !0
  %v11 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 4
  store float 0.000000e+00, float* %v11, align 16, !tbaa !0
  call void @f1(float* %v3) #2
  call void @llvm.lifetime.end.p0i8(i64 256, i8* %v1) #1
  ret void
}

declare void @f1(float*) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
