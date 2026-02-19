; RUN: llc -mtriple=hexagon -disable-hexagon-amodeopt < %s | FileCheck %s --check-prefix=CHECK-NO-AMODE1

; RUN: llc -mtriple=hexagon -disable-hexagon-amodeopt=0 < %s | FileCheck %s --check-prefix=CHECK-AMODE

; CHECK-NO-AMODE1: r{{[0-9]+}} = add([[REG1:(r[0-9]+)]],#{{[0-9]+}})
; CHECK-NO-AMODE1: r{{[0-9]+}} = add([[REG1]],#{{[0-9]+}})

; CHECK-AMODE: [[REG3:(r[0-9]+)]] = add(r{{[0-9]+}},#{{[0-9]+}})
; CHECK-AMODE: v{{.*}} = vmem([[REG3]]+#{{[0-9]+}})
; CHECK-AMODE: v{{.*}} = vmem([[REG3]]+#{{[0-9]+}})

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport void @foo() local_unnamed_addr #0 {
entry:
  %0 = load ptr, ptr undef, align 4
  %1 = bitcast ptr %0 to ptr
  %2 = or i32 undef, 128
  %3 = getelementptr half, ptr %1, i32 %2
  %4 = bitcast ptr %3 to ptr
  %5 = load ptr, ptr undef, align 4
  %6 = getelementptr i8, ptr %5, i32 1024
  %7 = bitcast ptr %6 to ptr
  %8 = load <64 x half>, ptr %7
  %9 = getelementptr i8, ptr %5, i32 1152
  %10 = bitcast ptr %9 to ptr
  %11 = load <64 x half>, ptr %10
  %12 = fadd <64 x half> %8, %11
  store <64 x half> %12, ptr %4, align 128
  call void @llvm.assume(i1 true) [ "align"(ptr undef, i32 128) ]
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: argmemonly nofree nosync nounwind readonly willreturn
declare <64 x half> @llvm.masked.load.v64f16.p0(ptr, i32 immarg, <64 x i1>, <64 x half>) #2

attributes #0 = { "target-features"="+hvxv68,+hvx-length128b,+hvx-qfloat" }
