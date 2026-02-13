; RUN: llc -O0 --verify-machineinstrs -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[#KERNEL:]] "kernel"
; CHECK: OpName %[[#FEATURE_PREDICATE_IDS:]] "llvm.amdgcn.feature.predicate.ids"
; CHECK: OpName %[[#SET_FPENV_I64:]] "spirv.llvm_set_fpenv_i64"
; CHECK: OpName %[[#ASHR_PK_I8_I32:]] "spirv.llvm_amdgcn_ashr_pk_i8_i32"
; CHECK: OpName %[[#S_SLEEP_VAR:]] "spirv.llvm_amdgcn_s_sleep_var"
; CHECK: OpName %[[#S_WAIT_EVENT_EXPORT_READY:]] "spirv.llvm_amdgcn_s_wait_event_export_ready"
; CHECK: OpName %[[#S_TTRACEDATA_IMM:]] "spirv.llvm_amdgcn_s_ttracedata_imm"
; CHECK: OpDecorate %[[#IS_GFX950:]] SpecId 6
; CHECK: OpDecorate %[[#IS_GFX950_1:]] SpecId 6
; CHECK: OpDecorate %[[#IS_GFX1201:]] SpecId 1
; CHECK: OpDecorate %[[#HAS_GFX12_INSTS:]] SpecId 5
; CHECK: OpDecorate %[[#IS_GFX906:]] SpecId 4
; CHECK: OpDecorate %[[#IS_GFX1010:]] SpecId 2
; CHECK: OpDecorate %[[#IS_GFX1101:]] SpecId 3
; CHECK: OpDecorate %[[#IS_GFX1101_1:]] SpecId 2
; CHECK: OpDecorate %[[#IS_GFX1201_1:]] SpecId 1
; CHECK: OpDecorate %[[#HAS_GFX11_INSTS:]] SpecId 0
; CHECK: OpDecorate %[[#HAS_GFX10_INSTS:]] SpecId 0
; CHECK: %[[#BOOL:]] = OpTypeBool
; CHECK: %[[#UCHAR:]] = OpTypeInt 8
; CHECK: %[[#FEATURE_PREDICATE_IDS_MAP_STRLEN:]] = OpConstant %[[#]] 99
; CHECK: %[[#FEATURE_PREDICATE_IDS_MAP_STRTY:]] = OpTypeArray %[[#UCHAR]] %[[#FEATURE_PREDICATE_IDS_MAP_STRLEN]]
; CHECK: %[[#FEATURE_PREDICATE_IDS_MAP_STRVAL:]] = OpConstantComposite %[[#FEATURE_PREDICATE_IDS_MAP_STRTY]]
; CHECK: %[[#FEATURE_PREDICATE_IDS]] = OpVariable %[[#]] CrossWorkgroup %[[#FEATURE_PREDICATE_IDS_MAP_STRVAL]]
; CHECK: %[[#IS_GFX950]] = OpSpecConstantFalse %[[#BOOL]]
; CHECK: %[[#IS_GFX1201]] = OpSpecConstantFalse %[[#BOOL]]
; CHECK: %[[#HAS_GFX12_INSTS]] = OpSpecConstantFalse %[[#BOOL]]
; CHECK: %[[#IS_GFX906]] = OpSpecConstantFalse %[[#BOOL]]
; CHECK: %[[#IS_GFX1010]] = OpSpecConstantFalse %[[#BOOL]]
; CHECK: %[[#IS_GFX1101]] = OpSpecConstantFalse %[[#BOOL]]
; CHECK: %[[#IS_GFX1101_1]] = OpSpecConstantFalse %[[#BOOL]]
; CHECK: %[[#IS_GFX1201_1]] = OpSpecConstantFalse %[[#BOOL]]
; CHECK: %[[#HAS_GFX11_INSTS]] = OpSpecConstantFalse %[[#BOOL]]
; CHECK: %[[#HAS_GFX10_INSTS]] = OpSpecConstantFalse %[[#BOOL]]

declare void @llvm.amdgcn.s.monitor.sleep(i16 immarg) addrspace(4)

declare void @llvm.amdgcn.s.sleep(i32 immarg) addrspace(4)

declare i1 @_Z20__spirv_SpecConstantib(i32, i1) addrspace(4)

declare i16 @llvm.amdgcn.ashr.pk.i8.i32(i32, i32, i32) addrspace(4) #3

declare void @llvm.set.fpenv.i64(i64) addrspace(4) #4

declare void @llvm.amdgcn.s.sleep.var(i32) addrspace(4) #5

declare void @llvm.amdgcn.s.wait.event.export.ready() addrspace(4) #5

declare void @llvm.amdgcn.s.ttracedata.imm(i16 immarg) addrspace(4) #6

@p = external addrspace(1) global i32
@g = external addrspace(1) constant i32

define void @kernel() addrspace(4) {
; CHECK-DAG: %[[#KERNEL]] = OpFunction %33 None %34 ; -- Begin function kernel
; CHECK-NEXT: %2 = OpLabel
; CHECK-NEXT: %99 = OpLoad %36 %74 Aligned 4
; CHECK-NEXT: OpBranchConditional %[[#IS_GFX950]] %4 %3
; CHECK-NEXT: %3 = OpLabel
; CHECK-NEXT: %100 = OpFunctionCall %33 %[[#SET_FPENV_I64]] %50
; CHECK-NEXT: OpBranch %5
; CHECK-NEXT: %4 = OpLabel
; CHECK-NEXT: %101 = OpFunctionCall %39 %[[#ASHR_PK_I8_I32]] %49 %49 %49
; CHECK-NEXT: OpBranch %5
; CHECK-NEXT: %5 = OpLabel
; CHECK-NEXT: OpBranchConditional %[[#IS_GFX950_1]] %7 %6
; CHECK-NEXT: %6 = OpLabel
; CHECK-NEXT: OpBranchConditional %[[#IS_GFX1201]] %7 %8
; CHECK-NEXT: %7 = OpLabel
; CHECK-NEXT: %102 = OpFunctionCall %33 %[[#S_SLEEP_VAR]] %99
; CHECK-NEXT: OpBranch %8
; CHECK-NEXT: %8 = OpLabel
; CHECK-NEXT: OpBranchConditional %[[#HAS_GFX12_INSTS]] %10 %9
; CHECK-NEXT: %9 = OpLabel
; CHECK-NEXT: %103 = OpFunctionCall %33 %[[#S_WAIT_EVENT_EXPORT_READY]]
; CHECK-NEXT: OpBranch %14
; CHECK-NEXT: %10 = OpLabel
; CHECK-NEXT: OpBranchConditional %[[#IS_GFX906]] %12 %11
; CHECK-NEXT: %11 = OpLabel
; CHECK-NEXT: OpBranchConditional %[[#IS_GFX1010]] %12 %13
; CHECK-NEXT: %12 = OpLabel
; CHECK-NEXT: %104 = OpFunctionCall %33 %[[#S_TTRACEDATA_IMM]] %48
; CHECK-NEXT: OpBranch %13
; CHECK-NEXT: %13 = OpLabel
; CHECK-NEXT: OpBranch %14
; CHECK-NEXT: %14 = OpLabel
; CHECK-NEXT: OpBranch %15
; CHECK-NEXT: %15 = OpLabel
; CHECK-NEXT: OpBranchConditional %[[#IS_GFX1101]] %16 %17
; CHECK-NEXT: %16 = OpLabel
; CHECK-NEXT: %105 = OpLoad %36 %86 Aligned 4
; CHECK-NEXT: %106 = OpIAdd %36 %105 %99
; CHECK-NEXT: OpStore %86 %106 Aligned 4
; CHECK-NEXT: OpBranch %17
; CHECK-NEXT: %17 = OpLabel
; CHECK-NEXT: OpBranch %18
; CHECK-NEXT: %18 = OpLabel
; CHECK-NEXT: %107 = OpLoad %36 %86 Aligned 4
; CHECK-NEXT: %108 = OpISub %36 %107 %99
; CHECK-NEXT: OpStore %86 %108 Aligned 4
; CHECK-NEXT: OpBranch %19
; CHECK-NEXT: %19 = OpLabel
; CHECK-NEXT: OpBranch %20
; CHECK-NEXT: %20 = OpLabel
; CHECK-NEXT: OpBranchConditional %[[#IS_GFX1101_1]] %21 %22
; CHECK-NEXT: %21 = OpLabel
; CHECK-NEXT: OpBranch %22
; CHECK-NEXT: %22 = OpLabel
; CHECK-NEXT: OpBranchConditional %[[#IS_GFX1201_1]] %26 %23
; CHECK-NEXT: %23 = OpLabel
; CHECK-NEXT: OpBranchConditional %[[#HAS_GFX11_INSTS]] %24 %25
; CHECK-NEXT: %24 = OpLabel
; CHECK-NEXT: %109 = OpFunctionCall %33 %[[#S_TTRACEDATA_IMM]] %48
; CHECK-NEXT: OpBranch %25
; CHECK-NEXT: %25 = OpLabel
; CHECK-NEXT: OpBranch %27
; CHECK-NEXT: %26 = OpLabel
; CHECK-NEXT: %110 = OpFunctionCall %33 %[[#S_WAIT_EVENT_EXPORT_READY]]
; CHECK-NEXT: OpBranch %27
; CHECK-NEXT: %27 = OpLabel
; CHECK-NEXT: OpBranch %28
; CHECK-NEXT: %28 = OpLabel
; CHECK-NEXT: %111 = OpLoad %36 %86 Aligned 4
; CHECK-NEXT: %112 = OpISub %36 %111 %99
; CHECK-NEXT: OpStore %86 %112 Aligned 4
; CHECK-NEXT: OpBranch %29
; CHECK-NEXT: %29 = OpLabel
; CHECK-NEXT: OpBranch %30
; CHECK-NEXT: %30 = OpLabel
; CHECK-NEXT: OpBranchConditional %[[#HAS_GFX10_INSTS]] %31 %32
; CHECK-NEXT: %31 = OpLabel
; CHECK-NEXT: OpBranch %32
; CHECK-NEXT: %32 = OpLabel

entry:
  %x = load i32, ptr addrspace(1) @g
  %is.gfx950. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !9
  br i1 %is.gfx950., label %cond.true, label %cond.false
cond.true:
  %0 = call addrspace(4) i16 @llvm.amdgcn.ashr.pk.i8.i32(i32 8, i32 8, i32 8)
  br label %cond.end
cond.false:
  call addrspace(4) void @llvm.set.fpenv.i64(i64 -1)
  br label %cond.end
cond.end:
  %is.gfx1201. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !9
  br i1 %is.gfx1201., label %if.then, label %lor.lhs.false
lor.lhs.false:
  %has.gfx12-insts. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !10
  br i1 %has.gfx12-insts., label %if.then, label %if.end
if.then:
  call addrspace(4) void @llvm.amdgcn.s.sleep.var(i32 %x)
  br label %if.end
if.end:
  %is.gfx906. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !11
  br i1 %is.gfx906., label %if.else, label %if.then2
if.then2:
  call addrspace(4) void @llvm.amdgcn.s.wait.event.export.ready()
  br label %if.end6
if.else:
  %is.gfx1010. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !12
  br i1 %is.gfx1010., label %if.then4, label %lor.lhs.false3
lor.lhs.false3:
  %is.gfx1101. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !13
  br i1 %is.gfx1101., label %if.then4, label %if.end5
if.then4:
  call addrspace(4) void @llvm.amdgcn.s.ttracedata.imm(i16 1)
  br label %if.end5
if.end5:
  br label %if.end6
if.end6:
  br label %while.cond
while.cond:
  %is.gfx1101.7 = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !14
  br i1 %is.gfx1101.7, label %while.body, label %while.end
while.body:
  %4 = load i32, ptr addrspace(1) @p
  %add = add i32 %4, %x
  store i32 %add, ptr addrspace(1) @p
  br label %while.end
while.end:
  br label %do.body
do.body:
  %7 = load i32, ptr addrspace(1) @p
  %sub = sub i32 %7, %x
  store i32 %sub, ptr addrspace(1) @p
  br label %do.end
do.cond:
  %is.gfx1010.8 = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !14
  br i1 %is.gfx1010.8, label %do.body, label %do.end
do.end:
  br label %for.cond
for.cond:
  %is.gfx1201.9 = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !13
  br i1 %is.gfx1201.9, label %for.body, label %for.end
for.body:
  br label %for.end
for.inc:
  %9 = load i32, ptr addrspace(1) @p
  %inc = add i32 %9, 1
  store i32 %inc, ptr addrspace(1) @p
  br label %for.cond
for.end:
  %has.gfx11-insts. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !10
  br i1 %has.gfx11-insts., label %if.then10, label %if.else11
if.then10:
  call addrspace(4) void @llvm.amdgcn.s.wait.event.export.ready()
  br label %if.end14
if.else11:
  %has.gfx10-insts. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !18
  br i1 %has.gfx10-insts., label %if.then12, label %if.end13
if.then12:
  call addrspace(4) void @llvm.amdgcn.s.ttracedata.imm(i16 1)
  br label %if.end13
if.end13:
  br label %if.end14
if.end14:
  br label %do.body15
do.body15:
  %12 = load i32, ptr addrspace(1) @p
  %sub16 = sub i32 %12, %x
  store i32 %sub16, ptr addrspace(1) @p
  br label %do.end18
do.cond17:
  %has.gfx1250-insts. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !20
  br i1 %has.gfx1250-insts., label %do.body15, label %do.end18
do.end18:
  br label %for.cond19
for.cond19:
  %has.gfx11-insts.20 = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !18
  br i1 %has.gfx11-insts.20, label %for.body21, label %for.end24
for.body21:
  br label %for.end24
for.inc22:
  %14 = load i32, ptr addrspace(1) @p
  %inc23 = add i32 %14, 1
  store i32 %inc23, ptr addrspace(1) @p
  br label %for.cond19
for.end24:
  ret void
}

!9 = !{!"is.gfx950"}
!10 = !{!"is.gfx1201"}
!11 = !{!"has.gfx12-insts"}
!12 = !{!"is.gfx906"}
!13 = !{!"is.gfx1010"}
!14 = !{!"is.gfx1101"}
!18 = !{!"has.gfx11-insts"}
!20 = !{!"has.gfx1250-insts"}
