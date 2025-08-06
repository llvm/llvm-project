; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpFOrdEqual
; CHECK-DAG: OpFOrdGreaterThan
; CHECK-DAG: OpFOrdGreaterThanEqual
; CHECK-DAG: OpFOrdLessThan
; CHECK-DAG: OpFOrdLessThanEqual
; CHECK-DAG: OpFOrdNotEqual
; CHECK-DAG: OpOrdered
; CHECK-DAG: OpFUnordEqual
; CHECK-DAG: OpFUnordGreaterThan
; CHECK-DAG: OpFUnordGreaterThanEqual
; CHECK-DAG: OpFUnordLessThan
; CHECK-DAG: OpFUnordLessThanEqual
; CHECK-DAG: OpFUnordNotEqual
; CHECK-DAG: OpUnordered

define dso_local spir_kernel void @test(float %a){
entry:
  %cmp = tail call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %a, metadata !"oeq", metadata !"fpexcept.strict") #2
  %cmp1 = tail call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %a, metadata !"ogt", metadata !"fpexcept.strict") #2
  %cmp2 = tail call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %a, metadata !"oge", metadata !"fpexcept.strict") #2
  %cmp3 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"olt", metadata !"fpexcept.strict") #2
  %cmp4 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ole", metadata !"fpexcept.strict") #2
  %cmp5 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"one", metadata !"fpexcept.strict") #2
  %cmp6 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ord", metadata !"fpexcept.strict") #2
  %cmp7 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ueq", metadata !"fpexcept.strict") #2
  %cmp8 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ugt", metadata !"fpexcept.strict") #2
  %cmp9 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"uge", metadata !"fpexcept.strict") #2
  %cmp10 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ult", metadata !"fpexcept.strict") #2
  %cmp11 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ule", metadata !"fpexcept.strict") #2
  %cmp12 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"une", metadata !"fpexcept.strict") #2
  %cmp13 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"uno", metadata !"fpexcept.strict") #2

  ; Chain all comparisons using 'or'
  %or1 = or i1 %cmp, %cmp1
  %or2 = or i1 %or1, %cmp2
  %or3 = or i1 %or2, %cmp3
  %or4 = or i1 %or3, %cmp4
  %or5 = or i1 %or4, %cmp5
  %or6 = or i1 %or5, %cmp6
  %or7 = or i1 %or6, %cmp7
  %or8 = or i1 %or7, %cmp8
  %or9 = or i1 %or8, %cmp9
  %or10 = or i1 %or9, %cmp10
  %or11 = or i1 %or10, %cmp11
  %or12 = or i1 %or11, %cmp12
  %or13 = or i1 %or12, %cmp13
  br i1 %or13, label %true_block, label %false_block
true_block:
  ret void
false_block:
  ret void
}
declare i1 @llvm.experimental.constrained.fcmps.f32(float, float, metadata, metadata) #1
declare i1 @llvm.experimental.constrained.fcmp.f32(float, float, metadata, metadata) #1
