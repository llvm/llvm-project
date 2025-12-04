; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_ALTERA_fpga_buffer_location %s -o %t.spt
; RUN: FileCheck %s --input-file=%t.spt
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_ALTERA_fpga_buffer_location %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability FPGABufferLocationALTERA
; CHECK-DAG: OpExtension "SPV_ALTERA_fpga_buffer_location"
; CHECK-DAG: OpName %[[#ARGA:]] "a"
; CHECK-DAG: OpName %[[#ARGB:]] "b"
; CHECK-DAG: OpName %[[#ARGC:]] "c"
; CHECK-DAG: OpName %[[#ARGD:]] "d"
; CHECK-DAG: OpName %[[#ARGE:]] "e"
; CHECK-NOT: OpDecorate %[[#ARGC]] BufferLocationALTERA -1
; CHECK-NOT: OpDecorate %[[#ARGC]] BufferLocationALTERA -1
; CHECK-DAG: OpDecorate %[[#ARGA]] BufferLocationALTERA 1
; CHECK-DAG: OpDecorate %[[#ARGB]] BufferLocationALTERA 2
; CHECK-NOT: OpDecorate %[[#ARGD]] BufferLocationALTERA -1
; CHECK-NOT: OpDecorate %[[#ARGE]] BufferLocationALTERA 3
; CHECK-DAG: OpDecorate %[[#]] BufferLocationALTERA 123456789

; CHECK-DAG: %[[#]] = OpFunction
; CHECK-DAG: %[[#ARGA]] = OpFunctionParameter %[[#]]
; CHECK-DAG: %[[#ARGB]] = OpFunctionParameter %[[#]]
; CHECK-DAG: %[[#ARGC]] = OpFunctionParameter %[[#]]
 
%struct.MyIP = type { ptr addrspace(4) }

@.str.4 = internal unnamed_addr addrspace(1) constant [19 x i8] c"{5921:\22123456789\22}\00"
@.str.1 = internal unnamed_addr addrspace(1) constant [9 x i8] c"main.cpp\00"

define spir_kernel void @test(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i32 %d, i32 %e) local_unnamed_addr !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_buffer_location !6
{
entry:
  ret void
}

define spir_kernel void @test.1(ptr addrspace(4) %a) #0
{
entry:
  %0 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %a, ptr addrspace(1) getelementptr inbounds ([19 x i8], ptr addrspace(1) @.str.4, i32 0, i32 0), ptr addrspace(1) getelementptr inbounds ([9 x i8], ptr addrspace(1) @.str.1, i32 0, i32 0), i32 7, ptr addrspace(1) null)
  store i8 0, ptr addrspace(4) %0, align 8
  ret void
}

$test.2 = comdat any
define weak_odr dso_local spir_kernel void @test.2(ptr addrspace(1) align 4 %arg_a) #0 comdat !kernel_arg_buffer_location !7 {
entry:
  %this.addr.i = alloca ptr addrspace(4), align 8
  %arg_a.addr = alloca ptr addrspace(1), align 8
  %MyIP = alloca %struct.MyIP, align 8
  %arg_a.addr.ascast = addrspacecast ptr %arg_a.addr to ptr addrspace(4)
  %MyIP.ascast = addrspacecast ptr %MyIP to ptr addrspace(4)
  store ptr addrspace(1) %arg_a, ptr addrspace(4) %arg_a.addr.ascast, align 8
  %a = getelementptr inbounds %struct.MyIP, ptr addrspace(4) %MyIP.ascast, i32 0, i32 0
  %0 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %a, ptr addrspace(1) getelementptr inbounds ([33 x i8], ptr addrspace(1) @.str.4, i32 0, i32 0), ptr addrspace(1) getelementptr inbounds ([9 x i8], ptr addrspace(1) @.str.1, i32 0, i32 0), i32 7, ptr addrspace(1) null)
  %b = load ptr addrspace(1), ptr addrspace(4) %arg_a.addr.ascast, align 8
  %1 = addrspacecast ptr addrspace(1) %b to ptr addrspace(4)
  store ptr addrspace(4) %1, ptr addrspace(4) %0, align 8
  %this.addr.ascast.i = addrspacecast ptr %this.addr.i to ptr addrspace(4)
  store ptr addrspace(4) %MyIP.ascast, ptr addrspace(4) %this.addr.ascast.i, align 8
  %this1.i = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast.i, align 8
  %a.i = getelementptr inbounds %struct.MyIP, ptr addrspace(4) %this1.i, i32 0, i32 0
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %a.i, ptr addrspace(1) getelementptr inbounds ([19 x i8], ptr addrspace(1) @.str.4, i32 0, i32 0), ptr addrspace(1) getelementptr inbounds ([9 x i8], ptr addrspace(1) @.str.1, i32 0, i32 0), i32 7, ptr addrspace(1) null)
  %3 = load ptr addrspace(4), ptr addrspace(4) %2, align 8
  %4 = load i32, ptr addrspace(4) %3, align 4
  %inc.i = add nsw i32 %4, 1
  store i32 %inc.i, ptr addrspace(4) %3, align 4
  ret void
}

define spir_kernel void @test.3(ptr addrspace(1) align 4 %arg_a, ptr %arg_b) {
entry:
  %this.addr.i = alloca ptr addrspace(4), align 8
  %arg_a.addr = alloca ptr addrspace(1), align 8
  %MyIP = alloca %struct.MyIP, align 8
  %arg_a.addr.ascast = addrspacecast ptr %arg_a.addr to ptr addrspace(4)
  %MyIP.ascast = addrspacecast ptr %MyIP to ptr addrspace(4)
  store ptr addrspace(1) %arg_a, ptr addrspace(4) %arg_a.addr.ascast, align 8
  %a = getelementptr inbounds %struct.MyIP, ptr addrspace(4) %MyIP.ascast, i32 0, i32 0
  %0 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %a, ptr addrspace(1) getelementptr inbounds ([19 x i8], ptr addrspace(1) @.str.4, i32 0, i32 0), ptr addrspace(1) getelementptr inbounds ([9 x i8], ptr addrspace(1) @.str.1, i32 0, i32 0), i32 7, ptr addrspace(1) null)
  call void @llvm.memcpy.p4.p0(ptr addrspace(4) %0, ptr %arg_b, i64 4, i1 false)
  ret void
}

declare ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1)) #1
declare void @llvm.memcpy.p4.p0(ptr addrspace(4) noalias captures(none) writeonly, ptr noalias captures(none) readonly, i64, i1 immarg) #2

!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!1}
!opencl.compiler.options = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 2, i32 0}
!1 = !{}
!2 = !{!""}
!3 = !{i32 1, i32 1, i32 1}
!4 = !{!"none", !"none", !"none"}
!5 = !{!"int*", !"float*", !"int*"}
!6 = !{i32 1, i32 2, i32 -1, i32 -1, i32 3}
!7 = !{i32 -1}
