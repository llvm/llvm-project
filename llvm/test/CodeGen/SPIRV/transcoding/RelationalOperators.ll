; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; ModuleID = 'RelationalOperators.cl'
source_filename = "RelationalOperators.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv32-unknown-unknown"

; CHECK-SPIRV: %[[bool:[0-9]+]] = OpTypeBool
; CHECK-SPIRV: %[[bool2:[0-9]+]] = OpTypeVector %[[bool]] 2

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpUGreaterThan %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testUGreaterThan(uint2 a, uint2 b, global int2 *res) {
;   res[0] = a > b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testUGreaterThan(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %cmp = icmp ugt <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpSGreaterThan %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testSGreaterThan(int2 a, int2 b, global int2 *res) {
;   res[0] = a > b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testSGreaterThan(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !11 !kernel_arg_base_type !12 !kernel_arg_type_qual !7 {
entry:
  %cmp = icmp sgt <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpUGreaterThanEqual %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testUGreaterThanEqual(uint2 a, uint2 b, global int2 *res) {
;   res[0] = a >= b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testUGreaterThanEqual(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %cmp = icmp uge <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpSGreaterThanEqual %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testSGreaterThanEqual(int2 a, int2 b, global int2 *res) {
;   res[0] = a >= b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testSGreaterThanEqual(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !11 !kernel_arg_base_type !12 !kernel_arg_type_qual !7 {
entry:
  %cmp = icmp sge <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpULessThan %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testULessThan(uint2 a, uint2 b, global int2 *res) {
;   res[0] = a < b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testULessThan(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %cmp = icmp ult <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpSLessThan %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testSLessThan(int2 a, int2 b, global int2 *res) {
;   res[0] = a < b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testSLessThan(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !11 !kernel_arg_base_type !12 !kernel_arg_type_qual !7 {
entry:
  %cmp = icmp slt <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpULessThanEqual %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testULessThanEqual(uint2 a, uint2 b, global int2 *res) {
;   res[0] = a <= b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testULessThanEqual(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %cmp = icmp ule <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpSLessThanEqual %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testSLessThanEqual(int2 a, int2 b, global int2 *res) {
;   res[0] = a <= b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testSLessThanEqual(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !11 !kernel_arg_base_type !12 !kernel_arg_type_qual !7 {
entry:
  %cmp = icmp sle <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpFOrdEqual %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testFOrdEqual(float2 a, float2 b, global int2 *res) {
;   res[0] = a == b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testFOrdEqual(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !13 !kernel_arg_base_type !14 !kernel_arg_type_qual !7 {
entry:
  %cmp = fcmp oeq <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpFUnordNotEqual %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testFUnordNotEqual(float2 a, float2 b, global int2 *res) {
;   res[0] = a != b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testFUnordNotEqual(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !13 !kernel_arg_base_type !14 !kernel_arg_type_qual !7 {
entry:
  %cmp = fcmp une <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpFOrdGreaterThan %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testFOrdGreaterThan(float2 a, float2 b, global int2 *res) {
;   res[0] = a > b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testFOrdGreaterThan(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !13 !kernel_arg_base_type !14 !kernel_arg_type_qual !7 {
entry:
  %cmp = fcmp ogt <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpFOrdGreaterThanEqual %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testFOrdGreaterThanEqual(float2 a, float2 b, global int2 *res) {
;   res[0] = a >= b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testFOrdGreaterThanEqual(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !13 !kernel_arg_base_type !14 !kernel_arg_type_qual !7 {
entry:
  %cmp = fcmp oge <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpFOrdLessThan %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testFOrdLessThan(float2 a, float2 b, global int2 *res) {
;   res[0] = a < b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testFOrdLessThan(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !13 !kernel_arg_base_type !14 !kernel_arg_type_qual !7 {
entry:
  %cmp = fcmp olt <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV-NEXT: %[[A:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV-NEXT: %[[B:[0-9]+]] = OpFunctionParameter %{{[0-9]+}}
; CHECK-SPIRV: %{{[0-9]+}} = OpFOrdLessThanEqual %[[bool2]] %[[A]] %[[B]]
; CHECK-SPIRV: OpFunctionEnd

; kernel void testFOrdLessThanEqual(float2 a, float2 b, global int2 *res) {
;   res[0] = a <= b;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define dso_local spir_kernel void @testFOrdLessThanEqual(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !13 !kernel_arg_base_type !14 !kernel_arg_type_qual !7 {
entry:
  %cmp = fcmp ole <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn writeonly "frame-pointer"="none" "min-legal-vector-width"="64" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git 881b6a009fb6e2dd5fb924524cd6eacd14148a08)"}
!3 = !{i32 0, i32 0, i32 1}
!4 = !{!"none", !"none", !"none"}
!5 = !{!"uint2", !"uint2", !"int2*"}
!6 = !{!"uint __attribute__((ext_vector_type(2)))", !"uint __attribute__((ext_vector_type(2)))", !"int __attribute__((ext_vector_type(2)))*"}
!7 = !{!"", !"", !""}
!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"int2", !"int2", !"int2*"}
!12 = !{!"int __attribute__((ext_vector_type(2)))", !"int __attribute__((ext_vector_type(2)))", !"int __attribute__((ext_vector_type(2)))*"}
!13 = !{!"float2", !"float2", !"int2*"}
!14 = !{!"float __attribute__((ext_vector_type(2)))", !"float __attribute__((ext_vector_type(2)))", !"int __attribute__((ext_vector_type(2)))*"}
