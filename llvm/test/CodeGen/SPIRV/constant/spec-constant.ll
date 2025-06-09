; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpDecorate [[bool_const:%[0-9]+]] SpecId 1
; CHECK-DAG: OpDecorate [[short_const:%[0-9]+]] SpecId 2
; CHECK-DAG: OpDecorate [[int_const:%[0-9]+]] SpecId 3
; CHECK-DAG: OpDecorate [[long_const:%[0-9]+]] SpecId 4
; CHECK-DAG: OpDecorate [[float_const:%[0-9]+]] SpecId 8
; CHECK-DAG: OpDecorate [[double_const:%[0-9]+]] SpecId 9
; CHECK-DAG: OpDecorate [[enum_const:%[0-9]+]] SpecId 10

; CHECK-DAG: [[bool_const]] = OpSpecConstantTrue {{%[0-9]+}}
; CHECK-DAG: [[short_const]] = OpSpecConstant {{%[0-9]+}} 4
; CHECK-DAG: [[int_const]] = OpSpecConstant {{%[0-9]+}} 5
; CHECK-DAG: [[long_const]] = OpSpecConstant {{%[0-9]+}} 8
; CHECK-DAG: [[float_const]] = OpSpecConstant {{%[0-9]+}} 1112014848
; CHECK-DAG: [[double_const]] = OpSpecConstant {{%[0-9]+}} 0 1079574528
; CHECK-DAG: [[enum_const]] = OpSpecConstant {{%[0-9]+}} 30

@_ZL10bool_const = internal addrspace(10) global i32 0, align 4
@_ZL11short_const = internal addrspace(10) global i16 0, align 2
@_ZL9int_const = internal addrspace(10) global i32 0, align 4
@_ZL10long_const = internal addrspace(10) global i64 0, align 8
@_ZL11float_const = internal addrspace(10) global float 0.000000e+00, align 4
@_ZL12double_const = internal addrspace(10) global double 0.000000e+00, align 8
@_ZL10enum_const = internal addrspace(10) global i32 0, align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define void @main() local_unnamed_addr #0 {
entry:
  %0 = tail call spir_func i1 @_Z20__spirv_SpecConstantib(i32 1, i1 true)
  %storedv.i.i = zext i1 %0 to i32
  store i32 %storedv.i.i, ptr addrspace(10) @_ZL10bool_const, align 4, !tbaa !3
  %1 = tail call ptr @llvm.invariant.start.p10(i64 4, ptr addrspace(10) @_ZL10bool_const)
  %2 = tail call spir_func i16 @_Z20__spirv_SpecConstantis(i32 2, i16 4)
  store i16 %2, ptr addrspace(10) @_ZL11short_const, align 2, !tbaa !7
  %3 = tail call ptr @llvm.invariant.start.p10(i64 2, ptr addrspace(10) @_ZL11short_const)
  %4 = tail call spir_func i32 @_Z20__spirv_SpecConstantii(i32 3, i32 5)
  store i32 %4, ptr addrspace(10) @_ZL9int_const, align 4, !tbaa !9
  %5 = tail call ptr @llvm.invariant.start.p10(i64 4, ptr addrspace(10) @_ZL9int_const)
  %6 = tail call spir_func i64 @_Z20__spirv_SpecConstantix(i32 4, i64 8)
  store i64 %6, ptr addrspace(10) @_ZL10long_const, align 8, !tbaa !11
  %7 = tail call ptr @llvm.invariant.start.p10(i64 8, ptr addrspace(10) @_ZL10long_const)
  %14 = tail call reassoc nnan ninf nsz arcp afn spir_func float @_Z20__spirv_SpecConstantif(i32 8, float 5.000000e+01)
  store float %14, ptr addrspace(10) @_ZL11float_const, align 4, !tbaa !13
  %15 = tail call ptr @llvm.invariant.start.p10(i64 4, ptr addrspace(10) @_ZL11float_const)
  %16 = tail call reassoc nnan ninf nsz arcp afn spir_func double @_Z20__spirv_SpecConstantid(i32 9, double 1.000000e+02)
  store double %16, ptr addrspace(10) @_ZL12double_const, align 8, !tbaa !15
  %17 = tail call ptr @llvm.invariant.start.p10(i64 8, ptr addrspace(10) @_ZL12double_const)
  %18 = tail call spir_func i32 @_Z20__spirv_SpecConstantii(i32 10, i32 30)
  store i32 %18, ptr addrspace(10) @_ZL10enum_const, align 4, !tbaa !17
  %19 = tail call ptr @llvm.invariant.start.p10(i64 4, ptr addrspace(10) @_ZL10enum_const)
  ret void
}


declare i1 @_Z20__spirv_SpecConstantib(i32, i1)
declare i8 @_Z20__spirv_SpecConstantia(i32, i8)
declare i16 @_Z20__spirv_SpecConstantis(i32, i16)
declare i32 @_Z20__spirv_SpecConstantii(i32, i32)
declare i64 @_Z20__spirv_SpecConstantix(i32, i64)
declare float @_Z20__spirv_SpecConstantif(i32, float)
declare double @_Z20__spirv_SpecConstantid(i32, double)

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) "approx-func-fp-math"="true" "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 21.0.0git (git@github.com:s-perron/llvm-project.git 060dd85ac525db3606a5ec7ca12ba346a7f075af)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"bool", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"short", !5, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !5, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"long long", !5, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"float", !5, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"double", !5, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"_ZTS1E", !5, i64 0}
