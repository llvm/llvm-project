; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_kernel_attributes,+SPV_INTEL_fpga_cluster_attributes,+SPV_INTEL_loop_fuse,+SPV_INTEL_fpga_invocation_pipelining_attributes %s -o - | FileCheck %s
; XFAIL: *

; CHECK: OpCapability FPGAKernelAttributesINTEL
; CHECK: OpCapability FPGAClusterAttributesINTEL
; CHECK: OpCapability LoopFuseINTEL
; CHECK: OpCapability FPGAInvocationPipeliningAttributesINTEL
; CHECK: OpExtension "SPV_INTEL_fpga_cluster_attributes"
; CHECK: OpExtension "SPV_INTEL_fpga_invocation_pipelining_attributes"
; CHECK: OpExtension "SPV_INTEL_loop_fuse"
; CHECK: OpEntryPoint Kernel %[[#FUNCENTRY2:]] "_ZTSZ3barvE11kernel_name3"
; CHECK: OpExecutionMode %[[#FUNCENTRY:]] MaxWorkgroupSizeINTEL 1 1 1
; CHECK: OpExecutionMode %[[#FUNCENTRY]] MaxWorkDimINTEL 1
; CHECK: OpExecutionMode %[[#FUNCENTRY]] NoGlobalOffsetINTEL
; CHECK: OpExecutionMode %[[#FUNCENTRY]] NumSIMDWorkitemsINTEL 8
; CHECK: OpExecutionMode %[[#FUNCENTRY]] SchedulerTargetFmaxMhzINTEL 1000
; CHECK-DAG: OpDecorate %[[#FUNCENTRY]] StallEnableINTEL
; CHECK-DAG: OpDecorate %[[#FUNCENTRY]] FuseLoopsInFunctionINTEL 3 1
; CHECK-DAG: OpDecorate %[[#FUNCENTRY]] InitiationIntervalINTEL 10
; CHECK-DAG: OpDecorate %[[#FUNCENTRY]] MaxConcurrencyINTEL 12
; CHECK-DAG: OpDecorate %[[#FUNCENTRY]] PipelineEnableINTEL 0
; CHECK: OpDecorate %[[#FUNCENTRY2]] PipelineEnableINTEL 1
; CHECK: %[[#FUNCENTRY]] = OpFunction
; CHECK: %[[#FUNCENTRY2]] = OpFunction

%class._ZTS3Foo.Foo = type { i8 }
%"class._ZTSZ3barvE3$_0.anon" = type { i8 }

$_ZN3FooclEv = comdat any

define spir_kernel void @_ZTSZ3barvE11kernel_name() !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 !num_simd_work_items !5 !max_work_group_size !6 !max_global_work_dim !7 !no_global_work_offset !4 !stall_enable !7 !scheduler_target_fmax_mhz !12 !loop_fuse !13 !initiation_interval !14 !max_concurrency !15 !pipeline_kernel !16 {
entry:
  %Foo = alloca %class._ZTS3Foo.Foo, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %Foo) #4
  %0 = addrspacecast ptr %Foo to ptr addrspace(4)
  call spir_func void @_ZN3FooclEv(ptr addrspace(4) %0)
  call void @llvm.lifetime.end.p0(i64 1, ptr %Foo) #4
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none))

define linkonce_odr spir_func void @_ZN3FooclEv(ptr addrspace(4) %this) comdat align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  store ptr addrspace(4) %this, ptr %this.addr, align 8, !tbaa !8
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  ret void
}

declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none))
define spir_kernel void @_ZTSZ3barvE12kernel_name2() !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %"class._ZTSZ3barvE3$_0.anon", align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %0)
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  call spir_func void @"_ZZ3barvENK3$_0clEv"(ptr addrspace(4) %1)
  call void @llvm.lifetime.end.p0(i64 1, ptr %0)
  ret void
}
define internal spir_func void @"_ZZ3barvENK3$_0clEv"(ptr addrspace(4) %this) align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  store ptr addrspace(4) %this, ptr %this.addr, align 8, !tbaa !8
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  ret void
}

define spir_kernel void @_ZTSZ3barvE11kernel_name3() !pipeline_kernel !7 {
entry:
  %Foo = alloca %class._ZTS3Foo.Foo, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %Foo)
  %0 = addrspacecast ptr %Foo to ptr addrspace(4)
  call spir_func void @_ZN3FooclEv(ptr addrspace(4) %0)
  call void @llvm.lifetime.end.p0(i64 1, ptr %Foo)
  ret void
}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 11.0.0"}
!4 = !{}
!5 = !{i32 8}
!6 = !{i32 1, i32 1, i32 1}
!7 = !{i32 1}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{i32 1000}
!13 = !{i32 3, i32 1}
!14 = !{i32 10}
!15 = !{i32 12}
!16 = !{i32 0}
