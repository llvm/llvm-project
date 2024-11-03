; RUN: llc -march=hexagon -disable-hexagon-amodeopt < %s | FileCheck %s --check-prefix=CHECK-NO-AMODE1

; RUN: llc -march=hexagon -disable-hexagon-amodeopt=0 < %s | FileCheck %s --check-prefix=CHECK-AMODE

; CHECK-NO-AMODE1: r{{[0-9]+}} = add([[REG1:(r[0-9]+)]],#{{[0-9]+}})
; CHECK-NO-AMODE1: r{{[0-9]+}} = add([[REG1]],#{{[0-9]+}})

; CHECK-AMODE: [[REG3:(r[0-9]+)]] = add(r{{[0-9]+}},#{{[0-9]+}})
; CHECK-AMODE: v{{.*}} = vmem([[REG3]]+#{{[0-9]+}})
; CHECK-AMODE: v{{.*}} = vmem([[REG3]]+#{{[0-9]+}})

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport void @foo() local_unnamed_addr #0 {
entry:
  %0 = load i8*, i8** undef, align 4
  %1 = bitcast i8* %0 to half*
  %2 = or i32 undef, 128
  %3 = getelementptr half, half* %1, i32 %2
  %4 = bitcast half* %3 to <64 x half>*
  %5 = load i8*, i8** undef, align 4
  %6 = getelementptr i8, i8* %5, i32 1024
  %7 = bitcast i8* %6 to <64 x half>*
  %8 = load <64 x half>, <64 x half>* %7
  %9 = getelementptr i8, i8* %5, i32 1152
  %10 = bitcast i8* %9 to <64 x half>*
  %11 = load <64 x half>, <64 x half>* %10
  %12 = fadd <64 x half> %8, %11
  store <64 x half> %12, <64 x half>* %4, align 128
  call void @llvm.assume(i1 true) [ "align"(i8* undef, i32 128) ]
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: argmemonly nofree nosync nounwind readonly willreturn
declare <64 x half> @llvm.masked.load.v64f16.p0v64f16(<64 x half>*, i32 immarg, <64 x i1>, <64 x half>) #2

attributes #0 = { "target-features"="+hvxv68,+hvx-length128b,+hvx-qfloat" }
