; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

; CHECK: OpDecorate [[NumWorkgroups:%[0-9]*]] BuiltIn NumWorkgroups
; CHECK: OpDecorate [[WorkgroupSize:%[0-9]*]] BuiltIn WorkgroupSize
; CHECK: OpDecorate [[WorkgroupId:%[0-9]*]] BuiltIn WorkgroupId
; CHECK: OpDecorate [[LocalInvocationId:%[0-9]*]] BuiltIn LocalInvocationId
; CHECK: OpDecorate [[GlobalInvocationId:%[0-9]*]] BuiltIn GlobalInvocationId
; CHECK: OpDecorate [[GlobalSize:%[0-9]*]] BuiltIn GlobalSize
; CHECK: OpDecorate [[GlobalOffset:%[0-9]*]] BuiltIn GlobalOffset
; CHECK: OpDecorate [[SubgroupSize:%[0-9]*]] BuiltIn SubgroupSize
; CHECK: OpDecorate [[SubgroupMaxSize:%[0-9]*]] BuiltIn SubgroupMaxSize
; CHECK: OpDecorate [[NumSubgroups:%[0-9]*]] BuiltIn NumSubgroups
; CHECK: OpDecorate [[SubgroupId:%[0-9]*]] BuiltIn SubgroupId
; CHECK: OpDecorate [[SubgroupLocalInvocationId:%[0-9]*]] BuiltIn SubgroupLocalInvocationId
; CHECK: [[I32:%[0-9]*]] = OpTypeInt 32 0
; CHECK: [[I64:%[0-9]*]] = OpTypeInt 64 0
; CHECK: [[I32PTR:%[0-9]*]] = OpTypePointer Input [[I32]]
; CHECK: [[I64V3:%[0-9]*]] = OpTypeVector [[I64]] 3
; CHECK: [[I64V3PTR:%[0-9]*]] = OpTypePointer Input [[I64V3]]
; CHECK: [[NumWorkgroups]] = OpVariable [[I64V3PTR]] Input
; CHECK: [[WorkgroupSize]] = OpVariable [[I64V3PTR]] Input
; CHECK: [[WorkgroupId]] = OpVariable [[I64V3PTR]] Input
; CHECK: [[LocalInvocationId]] = OpVariable [[I64V3PTR]] Input
; CHECK: [[GlobalInvocationId]] = OpVariable [[I64V3PTR]] Input
; CHECK: [[GlobalSize]] = OpVariable [[I64V3PTR]] Input
; CHECK: [[GlobalOffset]] = OpVariable [[I64V3PTR]] Input
; CHECK: [[SubgroupSize]] = OpVariable [[I32PTR]] Input
; CHECK: [[SubgroupMaxSize]] = OpVariable [[I32PTR]] Input
; CHECK: [[NumSubgroups]] = OpVariable [[I32PTR]] Input
; CHECK: [[SubgroupId]] = OpVariable [[I32PTR]] Input
; CHECK: [[SubgroupLocalInvocationId]] = OpVariable [[I32PTR]] Input

@G_spv_num_workgroups_0 = global i64 0
@G_spv_num_workgroups_1 = global i64 0
@G_spv_num_workgroups_2 = global i64 0
@G_spv_workgroup_size_0 = global i64 0
@G_spv_workgroup_size_1 = global i64 0
@G_spv_workgroup_size_2 = global i64 0
@G_spv_group_id_0 = global i64 0
@G_spv_group_id_1 = global i64 0
@G_spv_group_id_2 = global i64 0
@G_spv_thread_id_in_group_0 = global i64 0
@G_spv_thread_id_in_group_1 = global i64 0
@G_spv_thread_id_in_group_2 = global i64 0
@G_spv_thread_id_0 = global i64 0
@G_spv_thread_id_1 = global i64 0
@G_spv_thread_id_2 = global i64 0
@G_spv_global_size_0 = global i64 0
@G_spv_global_size_1 = global i64 0
@G_spv_global_size_2 = global i64 0
@G_spv_global_offset_0 = global i64 0
@G_spv_global_offset_1 = global i64 0
@G_spv_global_offset_2 = global i64 0

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func void @test_id_and_range() {
entry:
  %ssize = alloca i32, align 4
  %smax = alloca i32, align 4
  %snum = alloca i32, align 4
  %sid = alloca i32, align 4
  %sinvocid = alloca i32, align 4
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[NumWorkgroups]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 0
  %spv.num.workgroups = call i64 @llvm.spv.num.workgroups.i64(i32 0)
  store i64 %spv.num.workgroups, ptr @G_spv_num_workgroups_0
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[NumWorkgroups]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 1
  %spv.num.workgroups1 = call i64 @llvm.spv.num.workgroups.i64(i32 1)
  store i64 %spv.num.workgroups1, ptr @G_spv_num_workgroups_1
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[NumWorkgroups]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 2
  %spv.num.workgroups2 = call i64 @llvm.spv.num.workgroups.i64(i32 2)
  store i64 %spv.num.workgroups2, ptr @G_spv_num_workgroups_2
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[WorkgroupSize]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 0
  %spv.workgroup.size = call i64 @llvm.spv.workgroup.size.i64(i32 0)
  store i64 %spv.workgroup.size, ptr @G_spv_workgroup_size_0
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[WorkgroupSize]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 1
  %spv.workgroup.size3 = call i64 @llvm.spv.workgroup.size.i64(i32 1)
  store i64 %spv.workgroup.size3, ptr @G_spv_workgroup_size_1
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[WorkgroupSize]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 2
  %spv.workgroup.size4 = call i64 @llvm.spv.workgroup.size.i64(i32 2)
  store i64 %spv.workgroup.size4, ptr @G_spv_workgroup_size_2
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[WorkgroupId]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 0
  %spv.group.id = call i64 @llvm.spv.group.id.i64(i32 0)
  store i64 %spv.group.id, ptr @G_spv_group_id_0
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[WorkgroupId]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 1
  %spv.group.id5 = call i64 @llvm.spv.group.id.i64(i32 1)
  store i64 %spv.group.id5, ptr @G_spv_group_id_1
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[WorkgroupId]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 2
  %spv.group.id6 = call i64 @llvm.spv.group.id.i64(i32 2)
  store i64 %spv.group.id6, ptr @G_spv_group_id_2
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[LocalInvocationId]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 0
  %spv.thread.id.in.group = call i64 @llvm.spv.thread.id.in.group.i64(i32 0)
  store i64 %spv.thread.id.in.group, ptr @G_spv_thread_id_in_group_0
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[LocalInvocationId]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 1
  %spv.thread.id.in.group7 = call i64 @llvm.spv.thread.id.in.group.i64(i32 1)
  store i64 %spv.thread.id.in.group7, ptr @G_spv_thread_id_in_group_1
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[LocalInvocationId]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 2
  %spv.thread.id.in.group8 = call i64 @llvm.spv.thread.id.in.group.i64(i32 2)
  store i64 %spv.thread.id.in.group8, ptr @G_spv_thread_id_in_group_2
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[GlobalInvocationId]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 0
  %spv.thread.id = call i64 @llvm.spv.thread.id.i64(i32 0)
  store i64 %spv.thread.id, ptr @G_spv_thread_id_0
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[GlobalInvocationId]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 1
  %spv.thread.id9 = call i64 @llvm.spv.thread.id.i64(i32 1)
  store i64 %spv.thread.id9, ptr @G_spv_thread_id_1
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[GlobalInvocationId]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 2
  %spv.thread.id10 = call i64 @llvm.spv.thread.id.i64(i32 2)
  store i64 %spv.thread.id10, ptr @G_spv_thread_id_2
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[GlobalSize]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 0
  %spv.num.workgroups11 = call i64 @llvm.spv.global.size.i64(i32 0)
  store i64 %spv.num.workgroups11, ptr @G_spv_global_size_0
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[GlobalSize]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 1
  %spv.num.workgroups12 = call i64 @llvm.spv.global.size.i64(i32 1)
  store i64 %spv.num.workgroups12, ptr @G_spv_global_size_1
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[GlobalSize]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 2
  %spv.num.workgroups13 = call i64 @llvm.spv.global.size.i64(i32 2)
  store i64 %spv.num.workgroups13, ptr @G_spv_global_size_2
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[GlobalOffset]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 0
  %spv.global.offset = call i64 @llvm.spv.global.offset.i64(i32 0)
  store i64 %spv.global.offset, ptr @G_spv_global_offset_0
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[GlobalOffset]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 1
  %spv.global.offset14 = call i64 @llvm.spv.global.offset.i64(i32 1)
  store i64 %spv.global.offset14, ptr @G_spv_global_offset_1
; CHECK: [[LD:%[0-9]*]] = OpLoad [[I64V3]] [[GlobalOffset]]
; CHECK: OpCompositeExtract [[I64]] [[LD]] 2
  %spv.global.offset15 = call i64 @llvm.spv.global.offset.i64(i32 2)
  store i64 %spv.global.offset15, ptr @G_spv_global_offset_2
; CHECK: OpLoad %5 [[SubgroupSize]]
  %0 = call i32 @llvm.spv.subgroup.size()
  store i32 %0, ptr %ssize, align 4
; CHECK: OpLoad %5 [[SubgroupMaxSize]]
  %1 = call i32 @llvm.spv.subgroup.max.size()
  store i32 %1, ptr %smax, align 4
; CHECK: OpLoad %5 [[NumSubgroups]]
  %2 = call i32 @llvm.spv.num.subgroups()
  store i32 %2, ptr %snum, align 4
; CHECK: OpLoad %5 [[SubgroupId]]
  %3 = call i32 @llvm.spv.subgroup.id()
  store i32 %3, ptr %sid, align 4
; CHECK: OpLoad %5 [[SubgroupLocalInvocationId]]
  %4 = call i32 @llvm.spv.subgroup.local.invocation.id()
  store i32 %4, ptr %sinvocid, align 4
  ret void
}

declare i64 @llvm.spv.num.workgroups.i64(i32)
declare i64 @llvm.spv.workgroup.size.i64(i32)
declare i64 @llvm.spv.group.id.i64(i32)
declare i64 @llvm.spv.thread.id.in.group.i64(i32)
declare i64 @llvm.spv.thread.id.i64(i32)
declare i64 @llvm.spv.global.size.i64(i32)
declare i64 @llvm.spv.global.offset.i64(i32)
declare noundef i32 @llvm.spv.subgroup.size()
declare noundef i32 @llvm.spv.subgroup.max.size()
declare noundef i32 @llvm.spv.num.subgroups()
declare noundef i32 @llvm.spv.subgroup.id()
declare noundef i32 @llvm.spv.subgroup.local.invocation.id()
