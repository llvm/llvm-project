; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Kernel
; CHECK: OpCapability Addresses
; CHECK: OpCapability Pipes
; CHECK: OpCapability Int8
; CHECK: OpCapability GenericPointer

; CHECK-DAG: %[[#PipeWriteTy:]] = OpTypePipe WriteOnly
; CHECK-DAG: %[[#PipeReadTy:]] = OpTypePipe ReadOnly
; CHECK-DAG: %[[#ReserveIdTy:]] = OpTypeReserveId
; CHECK-DAG: %[[#BoolTy:]] = OpTypeBool
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Uint1:]] = OpConstant %[[#Int32Ty]] 1
; CHECK-DAG: %[[#Uint2:]] = OpConstant %[[#Int32Ty]] 2
; CHECK-DAG: %[[#Uint3:]] = OpConstant %[[#Int32Ty]] 3
; CHECK-DAG: %[[#Uint4:]] = OpConstant %[[#Int32Ty]] 4
; CHECK-DAG: %[[#NullUint:]] = OpConstantNull %[[#Int32Ty]]

; CHECK: OpFunction
; CHECK: %[[#FuncParam1:]] = OpFunctionParameter %[[#PipeWriteTy]]
; CHECK: %[[#FuncParam2:]] = OpFunctionParameter %[[#PipeReadTy]]

; CHECK: %[[#BasicWriteReserve:]] = OpReserveWritePipePackets %[[#ReserveIdTy]] %[[#FuncParam1]] %[[#Uint1]] %[[#Uint4]] %[[#Uint4]]
; CHECK: OpWritePipe %[[#Int32Ty]] %[[#FuncParam1]] %[[#]] %[[#Uint4]] %[[#Uint4]]
; CHECK: OpCommitWritePipe %[[#FuncParam1]] %[[#BasicWriteReserve]] %[[#Uint4]] %[[#Uint4]]
; CHECK: %[[#BasicReadReserve:]] = OpReserveReadPipePackets %[[#ReserveIdTy]] %[[#FuncParam2]] %[[#Uint1]] %[[#Uint4]] %[[#Uint4]]
; CHECK: OpReadPipe %[[#Int32Ty]] %[[#FuncParam2]] %[[#]] %[[#Uint4]] %[[#Uint4]]
; CHECK: OpCommitReadPipe %[[#FuncParam2]] %[[#BasicReadReserve]] %[[#Uint4]] %[[#Uint4]]

; --- Reserved pipe operations ---
; CHECK: %[[#ReservedWriteReserve:]] = OpReserveWritePipePackets %[[#ReserveIdTy]] %[[#FuncParam1]] %[[#Uint1]] %[[#Uint4]] %[[#Uint4]]
; CHECK: %[[#ReservedWrite:]] = OpReservedWritePipe %[[#Int32Ty]] %[[#FuncParam1]] %[[#ReservedWriteReserve]] %[[#NullUint]] %[[#]] %[[#Uint4]] %[[#Uint4]]
; CHECK: %[[#IsValidWrite:]] = OpIsValidReserveId %[[#BoolTy]] %[[#ReservedWriteReserve]]
; CHECK: OpCommitWritePipe %[[#FuncParam1]] %[[#ReservedWriteReserve]] %[[#Uint4]] %[[#Uint4]]
; CHECK: %[[#ReservedReadReserve:]] = OpReserveReadPipePackets %[[#ReserveIdTy]] %[[#FuncParam2]] %[[#Uint1]] %[[#Uint4]] %[[#Uint4]]
; CHECK: %[[#ReservedRead:]] = OpReservedReadPipe %[[#Int32Ty]] %[[#FuncParam2]] %[[#ReservedReadReserve]] %[[#NullUint]] %[[#]] %[[#Uint4]] %[[#Uint4]]
; CHECK: %[[#IsValidRead:]] = OpIsValidReserveId %[[#BoolTy]] %[[#ReservedReadReserve]]
; CHECK: OpCommitReadPipe %[[#FuncParam2]] %[[#ReservedReadReserve]] %[[#Uint4]] %[[#Uint4]]

; --- Pipe packet queries ---
; CHECK: %[[#MaxPacketsWO:]] = OpGetMaxPipePackets %[[#Int32Ty]] %[[#FuncParam1]] %[[#Uint4]] %[[#Uint4]]
; CHECK: OpStore %[[#]] %[[#MaxPacketsWO]] Aligned 4
; CHECK: %[[#NumPacketsWO:]] = OpGetNumPipePackets %[[#Int32Ty]] %[[#FuncParam1]] %[[#Uint4]] %[[#Uint4]]
; CHECK: OpStore %[[#]] %[[#NumPacketsWO]] Aligned 4
; CHECK: %[[#MaxPacketsRO:]] = OpGetMaxPipePackets %[[#Int32Ty]] %[[#FuncParam2]] %[[#Uint4]] %[[#Uint4]]
; CHECK: OpStore %[[#]] %[[#MaxPacketsRO]] Aligned 4
; CHECK: %[[#NumPacketsRO:]] = OpGetNumPipePackets %[[#Int32Ty]] %[[#FuncParam2]] %[[#Uint4]] %[[#Uint4]]
; CHECK: OpStore %[[#]] %[[#NumPacketsRO]] Aligned 4

; --- Workgroup operations ---
; CHECK: %[[#WorkgroupWriteReserve:]] = OpGroupReserveWritePipePackets %[[#ReserveIdTy]] %[[#Uint2]] %[[#FuncParam1]] %[[#Uint1]] %[[#Uint1]] %[[#Uint1]]
; CHECK: OpGroupCommitWritePipe %[[#Uint2]] %[[#FuncParam1]] %[[#WorkgroupWriteReserve]] %[[#Uint1]] %[[#Uint1]]
; CHECK: %[[#WorkgroupReadReserve:]] = OpGroupReserveReadPipePackets %[[#ReserveIdTy]] %[[#Uint2]] %[[#FuncParam2]] %[[#Uint1]] %[[#Uint1]] %[[#Uint1]]
; CHECK: OpGroupCommitReadPipe %[[#Uint2]] %[[#FuncParam2]] %[[#WorkgroupReadReserve]] %[[#Uint1]] %[[#Uint1]]

; --- Subgroup operations ---
; CHECK: %[[#SubgroupWriteReserve:]] = OpGroupReserveWritePipePackets %[[#ReserveIdTy]] %[[#Uint3]] %[[#FuncParam1]] %[[#Uint1]] %[[#Uint4]] %[[#Uint4]]
; CHECK: OpGroupCommitWritePipe %[[#Uint3]] %[[#FuncParam1]] %[[#SubgroupWriteReserve]] %[[#Uint4]] %[[#Uint4]]
; CHECK: %[[#SubgroupReadReserve:]] = OpGroupReserveReadPipePackets %[[#ReserveIdTy]] %[[#Uint3]] %[[#FuncParam2]] %[[#Uint1]] %[[#Uint4]] %[[#Uint4]]
; CHECK: OpGroupCommitReadPipe %[[#Uint3]] %[[#FuncParam2]] %[[#SubgroupReadReserve]] %[[#Uint4]] %[[#Uint4]]

define spir_kernel void @test_pipe_builtins(
  target("spirv.Pipe", 1) %out_pipe,
  target("spirv.Pipe", 0) %in_pipe,
  ptr addrspace(4) %src,
  ptr addrspace(4) %dst,
  ptr addrspace(1) %max_packets_wo,
  ptr addrspace(1) %num_packets_wo,
  ptr addrspace(1) %max_packets_ro,
  ptr addrspace(1) %num_packets_ro
) {
entry:
  ; Basic pipe operations
  %0 = call spir_func target("spirv.ReserveId") @__reserve_write_pipe(target("spirv.Pipe", 1) %out_pipe, i32 1, i32 4, i32 4)
  %1 = call spir_func i32 @__write_pipe_2(target("spirv.Pipe", 1) %out_pipe, ptr addrspace(4) %src, i32 4, i32 4)
  call spir_func void @__commit_write_pipe(target("spirv.Pipe", 1) %out_pipe, target("spirv.ReserveId") %0, i32 4, i32 4)
  
  %2 = call spir_func target("spirv.ReserveId") @__reserve_read_pipe(target("spirv.Pipe", 0) %in_pipe, i32 1, i32 4, i32 4)
  %3 = call spir_func i32 @__read_pipe_2(target("spirv.Pipe", 0) %in_pipe, ptr addrspace(4) %dst, i32 4, i32 4)
  call spir_func void @__commit_read_pipe(target("spirv.Pipe", 0) %in_pipe, target("spirv.ReserveId") %2, i32 4, i32 4)
  
  ; Reserved pipe operations
  %4 = call spir_func target("spirv.ReserveId") @__reserve_write_pipe(target("spirv.Pipe", 1) %out_pipe, i32 1, i32 4, i32 4)
  %5 = call spir_func i32 @__write_pipe_4(target("spirv.Pipe", 1) %out_pipe, target("spirv.ReserveId") %4, i32 0, ptr addrspace(4) %src, i32 4, i32 4)
  %6 = call spir_func i1 @_Z19is_valid_reserve_id13ocl_reserveid(target("spirv.ReserveId") %4)
  call spir_func void @__commit_write_pipe(target("spirv.Pipe", 1) %out_pipe, target("spirv.ReserveId") %4, i32 4, i32 4)
  
  %7 = call spir_func target("spirv.ReserveId") @__reserve_read_pipe(target("spirv.Pipe", 0) %in_pipe, i32 1, i32 4, i32 4)
  %8 = call spir_func i32 @__read_pipe_4(target("spirv.Pipe", 0) %in_pipe, target("spirv.ReserveId") %7, i32 0, ptr addrspace(4) %dst, i32 4, i32 4)
  %9 = call spir_func i1 @_Z19is_valid_reserve_id13ocl_reserveid(target("spirv.ReserveId") %7)
  call spir_func void @__commit_read_pipe(target("spirv.Pipe", 0) %in_pipe, target("spirv.ReserveId") %7, i32 4, i32 4)
  
  ; Pipe packet queries
  %10 = call spir_func i32 @__get_pipe_max_packets_wo(target("spirv.Pipe", 1) %out_pipe, i32 4, i32 4)
  store i32 %10, ptr addrspace(1) %max_packets_wo, align 4
  %11 = call spir_func i32 @__get_pipe_num_packets_wo(target("spirv.Pipe", 1) %out_pipe, i32 4, i32 4)
  store i32 %11, ptr addrspace(1) %num_packets_wo, align 4
  %12 = call spir_func i32 @__get_pipe_max_packets_ro(target("spirv.Pipe", 0) %in_pipe, i32 4, i32 4)
  store i32 %12, ptr addrspace(1) %max_packets_ro, align 4
  %13 = call spir_func i32 @__get_pipe_num_packets_ro(target("spirv.Pipe", 0) %in_pipe, i32 4, i32 4)
  store i32 %13, ptr addrspace(1) %num_packets_ro, align 4
  
  ; Workgroup operations
  %14 = call spir_func target("spirv.ReserveId") @__work_group_reserve_write_pipe(target("spirv.Pipe", 1) %out_pipe, i32 1, i32 1, i32 1)
  call spir_func void @__work_group_commit_write_pipe(target("spirv.Pipe", 1) %out_pipe, target("spirv.ReserveId") %14, i32 1, i32 1)
  %15 = call spir_func target("spirv.ReserveId") @__work_group_reserve_read_pipe(target("spirv.Pipe", 0) %in_pipe, i32 1, i32 1, i32 1)
  call spir_func void @__work_group_commit_read_pipe(target("spirv.Pipe", 0) %in_pipe, target("spirv.ReserveId") %15, i32 1, i32 1)
  
  ; Subgroup operations
  %16 = call spir_func target("spirv.ReserveId") @__sub_group_reserve_write_pipe(target("spirv.Pipe", 1) %out_pipe, i32 1, i32 4, i32 4)
  call spir_func void @__sub_group_commit_write_pipe(target("spirv.Pipe", 1) %out_pipe, target("spirv.ReserveId") %16, i32 4, i32 4)
  %17 = call spir_func target("spirv.ReserveId") @__sub_group_reserve_read_pipe(target("spirv.Pipe", 0) %in_pipe, i32 1, i32 4, i32 4)
  call spir_func void @__sub_group_commit_read_pipe(target("spirv.Pipe", 0) %in_pipe, target("spirv.ReserveId") %17, i32 4, i32 4)
  
  ret void
}

declare spir_func target("spirv.ReserveId") @__reserve_write_pipe(target("spirv.Pipe", 1), i32, i32, i32)
declare spir_func target("spirv.ReserveId") @__reserve_read_pipe(target("spirv.Pipe", 0), i32, i32, i32)
declare spir_func i32 @__write_pipe_2(target("spirv.Pipe", 1), ptr addrspace(4), i32, i32)
declare spir_func i32 @__read_pipe_2(target("spirv.Pipe", 0), ptr addrspace(4), i32, i32)
declare spir_func i32 @__write_pipe_4(target("spirv.Pipe", 1), target("spirv.ReserveId"), i32, ptr addrspace(4), i32, i32)
declare spir_func i32 @__read_pipe_4(target("spirv.Pipe", 0), target("spirv.ReserveId"), i32, ptr addrspace(4), i32, i32)
declare spir_func void @__commit_write_pipe(target("spirv.Pipe", 1), target("spirv.ReserveId"), i32, i32)
declare spir_func void @__commit_read_pipe(target("spirv.Pipe", 0), target("spirv.ReserveId"), i32, i32)
declare spir_func i1 @_Z19is_valid_reserve_id13ocl_reserveid(target("spirv.ReserveId"))
declare spir_func i32 @__get_pipe_max_packets_wo(target("spirv.Pipe", 1), i32, i32)
declare spir_func i32 @__get_pipe_num_packets_wo(target("spirv.Pipe", 1), i32, i32)
declare spir_func i32 @__get_pipe_max_packets_ro(target("spirv.Pipe", 0), i32, i32)
declare spir_func i32 @__get_pipe_num_packets_ro(target("spirv.Pipe", 0), i32, i32)
declare spir_func target("spirv.ReserveId") @__work_group_reserve_write_pipe(target("spirv.Pipe", 1), i32, i32, i32)
declare spir_func void @__work_group_commit_write_pipe(target("spirv.Pipe", 1), target("spirv.ReserveId"), i32, i32)
declare spir_func target("spirv.ReserveId") @__work_group_reserve_read_pipe(target("spirv.Pipe", 0), i32, i32, i32)
declare spir_func void @__work_group_commit_read_pipe(target("spirv.Pipe", 0), target("spirv.ReserveId"), i32, i32)
declare spir_func target("spirv.ReserveId") @__sub_group_reserve_write_pipe(target("spirv.Pipe", 1), i32, i32, i32)
declare spir_func void @__sub_group_commit_write_pipe(target("spirv.Pipe", 1), target("spirv.ReserveId"), i32, i32)
declare spir_func target("spirv.ReserveId") @__sub_group_reserve_read_pipe(target("spirv.Pipe", 0), i32, i32, i32)
declare spir_func void @__sub_group_commit_read_pipe(target("spirv.Pipe", 0), target("spirv.ReserveId"), i32, i32)
