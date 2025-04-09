; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; Test generated from the opencl test in the SPIRV-LLVM Translator (pipe_builtins.cl)

; CHECK: OpCapability Kernel
; CHECK: OpCapability Addresses
; CHECK: OpCapability Pipes
; CHECK: OpCapability Int64

; --- Check common types ---
; CHECK-DAG: %[[#PipeWriteTy:]] = OpTypePipe WriteOnly
; CHECK-DAG: %[[#PipeReadTy:]] = OpTypePipe ReadOnly
; CHECK-DAG: %[[#ReserveIdTy:]] = OpTypeReserveId
; CHECK-DAG: %[[#BoolTy:]] = OpTypeBool
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#CONST:]] = OpConstant %[[#Int32Ty]] 4
; CHECK-DAG: %[[#CONST2:]] = OpConstant %[[#Int32Ty]] 2
; CHECK-DAG: %[[#CONST3:]] = OpConstant %[[#Int32Ty]] 3
; CHECK-DAG: %[[#CONST4:]] = OpConstant %[[#Int32Ty]] 1

; --- Function: test_pipe_convenience_write_uint ---
; CHECK: %[[#ReadWriteInst1:]] = OpWritePipe %[[#Int32Ty]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
  
; --- Function: test_pipe_convenience_read_uint ---
; CHECK: %[[#ReadPipeInst2:]] = OpReadPipe %[[#Int32Ty]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
  
; --- Function: test_pipe_write ---
; CHECK: %[[#ReserveWrite1:]] = OpReserveWritePipePackets %[[#ReserveIdTy]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
; CHECK: %[[#IsValidWrite1:]] = OpIsValidReserveId %[[#BoolTy]] %[[#]]
; CHECK: %[[#ReservedWrite1:]] = OpReservedWritePipe %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
; CHECK: OpCommitWritePipe %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
  
; --- Function: test_pipe_query_functions ---
; CHECK: %[[#]] = OpGetMaxPipePackets %[[#Int32Ty]] %[[#]] %[[#CONST]] %[[#CONST]]
; CHECK: %[[#]] = OpGetNumPipePackets %[[#Int32Ty]] %[[#]] %[[#CONST]] %[[#CONST]]
  
; --- Function: test_pipe_read ---
; CHECK: %[[#ReserveRead1:]] = OpReserveReadPipePackets %[[#ReserveIdTy]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
; CHECK: %[[#IsValidRead1:]] = OpIsValidReserveId %[[#BoolTy]] %[[#]]
; CHECK: %[[#ReservedRead1:]] = OpReservedReadPipe %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
; CHECK: OpCommitReadPipe %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
  
; --- Function: test_pipe_workgroup_write_char ---
; CHECK: %[[#GRW1:]] = OpGroupReserveWritePipePackets %[[#ReserveIdTy]] %[[#CONST2]] %[[#]] %[[#]] %[[#CONST4]] %[[#CONST4]]
; CHECK: %[[#IsValidGRW1:]] = OpIsValidReserveId %[[#BoolTy]] %[[#]]
; CHECK: %[[#ReservedGWWrite:]] = OpReservedWritePipe %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#CONST4]] %[[#CONST4]]
; CHECK: OpGroupCommitWritePipe %[[#CONST2]] %[[#]] %[[#]] %[[#CONST4]] %[[#CONST4]]
  
; --- Function: test_pipe_workgroup_read_char ---
; CHECK: %[[#GRR1:]] = OpGroupReserveReadPipePackets %[[#ReserveIdTy]] %[[#CONST2]] %[[#]] %[[#]] %[[#CONST4]] %[[#CONST4]]
; CHECK: %[[#IsValidGRR1:]] = OpIsValidReserveId %[[#BoolTy]] %[[#]]
; CHECK: %[[#ReservedGWRead:]] = OpReservedReadPipe %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#CONST4]] %[[#CONST4]]
; CHECK: OpGroupCommitReadPipe %[[#CONST2]] %[[#]] %[[#]] %[[#CONST4]] %[[#CONST4]]
  
; --- Function: test_pipe_subgroup_write_uint ---
; CHECK: %[[#SRW1:]] = OpGroupReserveWritePipePackets %[[#ReserveIdTy]] %[[#CONST3]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
; CHECK: %[[#IsValidSRW1:]] = OpIsValidReserveId %[[#BoolTy]] %[[#]]
; CHECK: %[[#ReservedSWWrite:]] = OpReservedWritePipe %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
; CHECK: OpGroupCommitWritePipe %[[#CONST3]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
  
; --- Function: test_pipe_subgroup_read_uint ---
; CHECK: %[[#SRR1:]] = OpGroupReserveReadPipePackets %[[#ReserveIdTy]] %[[#CONST3]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
; CHECK: %[[#IsValidSRR1:]] = OpIsValidReserveId %[[#BoolTy]] %[[#]]
; CHECK: %[[#ReservedSWRead:]] = OpReservedReadPipe %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]
; CHECK: OpGroupCommitReadPipe %[[#CONST3]] %[[#]] %[[#]] %[[#CONST]] %[[#CONST]]

@test_pipe_workgroup_write_char.res_id = internal addrspace(3) global target("spirv.ReserveId") undef, align 8
@test_pipe_workgroup_read_char.res_id = internal addrspace(3) global target("spirv.ReserveId") undef, align 8

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_pipe_convenience_write_uint(ptr addrspace(1) noundef align 4 %src, target("spirv.Pipe", 1) %out_pipe) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %src.addr = alloca ptr addrspace(1), align 8
  %out_pipe.addr = alloca target("spirv.Pipe", 1), align 8
  %gid = alloca i32, align 4
  store ptr addrspace(1) %src, ptr %src.addr, align 8
  store target("spirv.Pipe", 1) %out_pipe, ptr %out_pipe.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %gid, align 4
  %0 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %1 = load ptr addrspace(1), ptr %src.addr, align 8
  %2 = load i32, ptr %gid, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %idxprom
  %3 = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  %4 = call spir_func i32 @__write_pipe_2(target("spirv.Pipe", 1) %0, ptr addrspace(4) %3, i32 4, i32 4)
  ret void
}

; Function Attrs: convergent nounwind willreturn memory(none)
declare spir_func i64 @_Z13get_global_idj(i32 noundef) #1

declare spir_func i32 @__write_pipe_2(target("spirv.Pipe", 1), ptr addrspace(4), i32, i32)

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_pipe_convenience_read_uint(target("spirv.Pipe", 0) %in_pipe, ptr addrspace(1) noundef align 4 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
entry:
  %in_pipe.addr = alloca target("spirv.Pipe", 0), align 8
  %dst.addr = alloca ptr addrspace(1), align 8
  %gid = alloca i32, align 4
  store target("spirv.Pipe", 0) %in_pipe, ptr %in_pipe.addr, align 8
  store ptr addrspace(1) %dst, ptr %dst.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %gid, align 4
  %0 = load target("spirv.Pipe", 0), ptr %in_pipe.addr, align 8
  %1 = load ptr addrspace(1), ptr %dst.addr, align 8
  %2 = load i32, ptr %gid, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %idxprom
  %3 = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  %4 = call spir_func i32 @__read_pipe_2(target("spirv.Pipe", 0) %0, ptr addrspace(4) %3, i32 4, i32 4)
  ret void
}

declare spir_func i32 @__read_pipe_2(target("spirv.Pipe", 0), ptr addrspace(4), i32, i32)

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_pipe_write(ptr addrspace(1) noundef align 4 %src, target("spirv.Pipe", 1) %out_pipe) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !6 {
entry:
  %src.addr = alloca ptr addrspace(1), align 8
  %out_pipe.addr = alloca target("spirv.Pipe", 1), align 8
  %gid = alloca i32, align 4
  %res_id = alloca target("spirv.ReserveId"), align 8
  store ptr addrspace(1) %src, ptr %src.addr, align 8
  store target("spirv.Pipe", 1) %out_pipe, ptr %out_pipe.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %gid, align 4
  %0 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %1 = call spir_func target("spirv.ReserveId") @__reserve_write_pipe(target("spirv.Pipe", 1) %0, i32 1, i32 4, i32 4)
  store target("spirv.ReserveId") %1, ptr %res_id, align 8
  %2 = load target("spirv.ReserveId"), ptr %res_id, align 8
  %call1 = call spir_func zeroext i1 @_Z19is_valid_reserve_id13ocl_reserveid(target("spirv.ReserveId") %2) #4
  br i1 %call1, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %3 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %4 = load target("spirv.ReserveId"), ptr %res_id, align 8
  %5 = load ptr addrspace(1), ptr %src.addr, align 8
  %6 = load i32, ptr %gid, align 4
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %idxprom
  %7 = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  %8 = call spir_func i32 @__write_pipe_4(target("spirv.Pipe", 1) %3, target("spirv.ReserveId") %4, i32 0, ptr addrspace(4) %7, i32 4, i32 4)
  %9 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %10 = load target("spirv.ReserveId"), ptr %res_id, align 8
  call spir_func void @__commit_write_pipe(target("spirv.Pipe", 1) %9, target("spirv.ReserveId") %10, i32 4, i32 4)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare spir_func target("spirv.ReserveId") @__reserve_write_pipe(target("spirv.Pipe", 1), i32, i32, i32)

; Function Attrs: convergent nounwind
declare spir_func zeroext i1 @_Z19is_valid_reserve_id13ocl_reserveid(target("spirv.ReserveId")) #2

declare spir_func i32 @__write_pipe_4(target("spirv.Pipe", 1), target("spirv.ReserveId"), i32, ptr addrspace(4), i32, i32)

declare spir_func void @__commit_write_pipe(target("spirv.Pipe", 1), target("spirv.ReserveId"), i32, i32)

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_pipe_query_functions(target("spirv.Pipe", 1) %out_pipe, ptr addrspace(1) noundef align 4 %num_packets, ptr addrspace(1) noundef align 4 %max_packets) #0 !kernel_arg_addr_space !11 !kernel_arg_access_qual !12 !kernel_arg_type !13 !kernel_arg_base_type !13 !kernel_arg_type_qual !14 {
entry:
  %out_pipe.addr = alloca target("spirv.Pipe", 1), align 8
  %num_packets.addr = alloca ptr addrspace(1), align 8
  %max_packets.addr = alloca ptr addrspace(1), align 8
  store target("spirv.Pipe", 1) %out_pipe, ptr %out_pipe.addr, align 8
  store ptr addrspace(1) %num_packets, ptr %num_packets.addr, align 8
  store ptr addrspace(1) %max_packets, ptr %max_packets.addr, align 8
  %0 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %1 = call spir_func i32 @__get_pipe_max_packets_wo(target("spirv.Pipe", 1) %0, i32 4, i32 4)
  %2 = load ptr addrspace(1), ptr %max_packets.addr, align 8
  store i32 %1, ptr addrspace(1) %2, align 4
  %3 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %4 = call spir_func i32 @__get_pipe_num_packets_wo(target("spirv.Pipe", 1) %3, i32 4, i32 4)
  %5 = load ptr addrspace(1), ptr %num_packets.addr, align 8
  store i32 %4, ptr addrspace(1) %5, align 4
  ret void
}

declare spir_func i32 @__get_pipe_max_packets_wo(target("spirv.Pipe", 1), i32, i32)

declare spir_func i32 @__get_pipe_num_packets_wo(target("spirv.Pipe", 1), i32, i32)

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_pipe_read(target("spirv.Pipe", 0) %in_pipe, ptr addrspace(1) noundef align 4 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !7 !kernel_arg_type !15 !kernel_arg_base_type !15 !kernel_arg_type_qual !9 {
entry:
  %in_pipe.addr = alloca target("spirv.Pipe", 0), align 8
  %dst.addr = alloca ptr addrspace(1), align 8
  %gid = alloca i32, align 4
  %res_id = alloca target("spirv.ReserveId"), align 8
  store target("spirv.Pipe", 0) %in_pipe, ptr %in_pipe.addr, align 8
  store ptr addrspace(1) %dst, ptr %dst.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %gid, align 4
  %0 = load target("spirv.Pipe", 0), ptr %in_pipe.addr, align 8
  %1 = call spir_func target("spirv.ReserveId") @__reserve_read_pipe(target("spirv.Pipe", 0) %0, i32 1, i32 4, i32 4)
  store target("spirv.ReserveId") %1, ptr %res_id, align 8
  %2 = load target("spirv.ReserveId"), ptr %res_id, align 8
  %call1 = call spir_func zeroext i1 @_Z19is_valid_reserve_id13ocl_reserveid(target("spirv.ReserveId") %2) #4
  br i1 %call1, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %3 = load target("spirv.Pipe", 0), ptr %in_pipe.addr, align 8
  %4 = load target("spirv.ReserveId"), ptr %res_id, align 8
  %5 = load ptr addrspace(1), ptr %dst.addr, align 8
  %6 = load i32, ptr %gid, align 4
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %idxprom
  %7 = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  %8 = call spir_func i32 @__read_pipe_4(target("spirv.Pipe", 0) %3, target("spirv.ReserveId") %4, i32 0, ptr addrspace(4) %7, i32 4, i32 4)
  %9 = load target("spirv.Pipe", 0), ptr %in_pipe.addr, align 8
  %10 = load target("spirv.ReserveId"), ptr %res_id, align 8
  call spir_func void @__commit_read_pipe(target("spirv.Pipe", 0) %9, target("spirv.ReserveId") %10, i32 4, i32 4)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare spir_func target("spirv.ReserveId") @__reserve_read_pipe(target("spirv.Pipe", 0), i32, i32, i32)

declare spir_func i32 @__read_pipe_4(target("spirv.Pipe", 0), target("spirv.ReserveId"), i32, ptr addrspace(4), i32, i32)

declare spir_func void @__commit_read_pipe(target("spirv.Pipe", 0), target("spirv.ReserveId"), i32, i32)

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_pipe_workgroup_write_char(ptr addrspace(1) noundef align 1 %src, target("spirv.Pipe", 1) %out_pipe) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !16 !kernel_arg_base_type !16 !kernel_arg_type_qual !6 {
entry:
  %src.addr = alloca ptr addrspace(1), align 8
  %out_pipe.addr = alloca target("spirv.Pipe", 1), align 8
  %gid = alloca i32, align 4
  store ptr addrspace(1) %src, ptr %src.addr, align 8
  store target("spirv.Pipe", 1) %out_pipe, ptr %out_pipe.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %gid, align 4
  %0 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %call1 = call spir_func i64 @_Z14get_local_sizej(i32 noundef 0) #3
  %1 = trunc i64 %call1 to i32
  %2 = call spir_func target("spirv.ReserveId") @__work_group_reserve_write_pipe(target("spirv.Pipe", 1) %0, i32 %1, i32 1, i32 1)
  store target("spirv.ReserveId") %2, ptr addrspace(3) @test_pipe_workgroup_write_char.res_id, align 8
  %3 = load target("spirv.ReserveId"), ptr addrspace(3) @test_pipe_workgroup_write_char.res_id, align 8
  %call2 = call spir_func zeroext i1 @_Z19is_valid_reserve_id13ocl_reserveid(target("spirv.ReserveId") %3) #4
  br i1 %call2, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %4 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %5 = load target("spirv.ReserveId"), ptr addrspace(3) @test_pipe_workgroup_write_char.res_id, align 8
  %call3 = call spir_func i64 @_Z12get_local_idj(i32 noundef 0) #3
  %6 = load ptr addrspace(1), ptr %src.addr, align 8
  %7 = load i32, ptr %gid, align 4
  %idxprom = sext i32 %7 to i64
  %arrayidx = getelementptr inbounds i8, ptr addrspace(1) %6, i64 %idxprom
  %8 = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  %9 = trunc i64 %call3 to i32
  %10 = call spir_func i32 @__write_pipe_4(target("spirv.Pipe", 1) %4, target("spirv.ReserveId") %5, i32 %9, ptr addrspace(4) %8, i32 1, i32 1)
  %11 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %12 = load target("spirv.ReserveId"), ptr addrspace(3) @test_pipe_workgroup_write_char.res_id, align 8
  call spir_func void @__work_group_commit_write_pipe(target("spirv.Pipe", 1) %11, target("spirv.ReserveId") %12, i32 1, i32 1)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: convergent nounwind willreturn memory(none)
declare spir_func i64 @_Z14get_local_sizej(i32 noundef) #1

declare spir_func target("spirv.ReserveId") @__work_group_reserve_write_pipe(target("spirv.Pipe", 1), i32, i32, i32)

; Function Attrs: convergent nounwind willreturn memory(none)
declare spir_func i64 @_Z12get_local_idj(i32 noundef) #1

declare spir_func void @__work_group_commit_write_pipe(target("spirv.Pipe", 1), target("spirv.ReserveId"), i32, i32)

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_pipe_workgroup_read_char(target("spirv.Pipe", 0) %in_pipe, ptr addrspace(1) noundef align 1 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !7 !kernel_arg_type !17 !kernel_arg_base_type !17 !kernel_arg_type_qual !9 {
entry:
  %in_pipe.addr = alloca target("spirv.Pipe", 0), align 8
  %dst.addr = alloca ptr addrspace(1), align 8
  %gid = alloca i32, align 4
  store target("spirv.Pipe", 0) %in_pipe, ptr %in_pipe.addr, align 8
  store ptr addrspace(1) %dst, ptr %dst.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %gid, align 4
  %0 = load target("spirv.Pipe", 0), ptr %in_pipe.addr, align 8
  %call1 = call spir_func i64 @_Z14get_local_sizej(i32 noundef 0) #3
  %1 = trunc i64 %call1 to i32
  %2 = call spir_func target("spirv.ReserveId") @__work_group_reserve_read_pipe(target("spirv.Pipe", 0) %0, i32 %1, i32 1, i32 1)
  store target("spirv.ReserveId") %2, ptr addrspace(3) @test_pipe_workgroup_read_char.res_id, align 8
  %3 = load target("spirv.ReserveId"), ptr addrspace(3) @test_pipe_workgroup_read_char.res_id, align 8
  %call2 = call spir_func zeroext i1 @_Z19is_valid_reserve_id13ocl_reserveid(target("spirv.ReserveId") %3) #4
  br i1 %call2, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %4 = load target("spirv.Pipe", 0), ptr %in_pipe.addr, align 8
  %5 = load target("spirv.ReserveId"), ptr addrspace(3) @test_pipe_workgroup_read_char.res_id, align 8
  %call3 = call spir_func i64 @_Z12get_local_idj(i32 noundef 0) #3
  %6 = load ptr addrspace(1), ptr %dst.addr, align 8
  %7 = load i32, ptr %gid, align 4
  %idxprom = sext i32 %7 to i64
  %arrayidx = getelementptr inbounds i8, ptr addrspace(1) %6, i64 %idxprom
  %8 = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  %9 = trunc i64 %call3 to i32
  %10 = call spir_func i32 @__read_pipe_4(target("spirv.Pipe", 0) %4, target("spirv.ReserveId") %5, i32 %9, ptr addrspace(4) %8, i32 1, i32 1)
  %11 = load target("spirv.Pipe", 0), ptr %in_pipe.addr, align 8
  %12 = load target("spirv.ReserveId"), ptr addrspace(3) @test_pipe_workgroup_read_char.res_id, align 8
  call spir_func void @__work_group_commit_read_pipe(target("spirv.Pipe", 0) %11, target("spirv.ReserveId") %12, i32 1, i32 1)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare spir_func target("spirv.ReserveId") @__work_group_reserve_read_pipe(target("spirv.Pipe", 0), i32, i32, i32)

declare spir_func void @__work_group_commit_read_pipe(target("spirv.Pipe", 0), target("spirv.ReserveId"), i32, i32)

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_pipe_subgroup_write_uint(ptr addrspace(1) noundef align 4 %src, target("spirv.Pipe", 1) %out_pipe) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %src.addr = alloca ptr addrspace(1), align 8
  %out_pipe.addr = alloca target("spirv.Pipe", 1), align 8
  %gid = alloca i32, align 4
  %res_id = alloca target("spirv.ReserveId"), align 8
  store ptr addrspace(1) %src, ptr %src.addr, align 8
  store target("spirv.Pipe", 1) %out_pipe, ptr %out_pipe.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %gid, align 4
  %0 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %call1 = call spir_func i32 @_Z18get_sub_group_sizev() #4
  %1 = call spir_func target("spirv.ReserveId") @__sub_group_reserve_write_pipe(target("spirv.Pipe", 1) %0, i32 %call1, i32 4, i32 4)
  store target("spirv.ReserveId") %1, ptr %res_id, align 8
  %2 = load target("spirv.ReserveId"), ptr %res_id, align 8
  %call2 = call spir_func zeroext i1 @_Z19is_valid_reserve_id13ocl_reserveid(target("spirv.ReserveId") %2) #4
  br i1 %call2, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %3 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %4 = load target("spirv.ReserveId"), ptr %res_id, align 8
  %call3 = call spir_func i32 @_Z22get_sub_group_local_idv() #4
  %5 = load ptr addrspace(1), ptr %src.addr, align 8
  %6 = load i32, ptr %gid, align 4
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %idxprom
  %7 = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  %8 = call spir_func i32 @__write_pipe_4(target("spirv.Pipe", 1) %3, target("spirv.ReserveId") %4, i32 %call3, ptr addrspace(4) %7, i32 4, i32 4)
  %9 = load target("spirv.Pipe", 1), ptr %out_pipe.addr, align 8
  %10 = load target("spirv.ReserveId"), ptr %res_id, align 8
  call spir_func void @__sub_group_commit_write_pipe(target("spirv.Pipe", 1) %9, target("spirv.ReserveId") %10, i32 4, i32 4)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func i32 @_Z18get_sub_group_sizev() #2

declare spir_func target("spirv.ReserveId") @__sub_group_reserve_write_pipe(target("spirv.Pipe", 1), i32, i32, i32)

; Function Attrs: convergent nounwind
declare spir_func i32 @_Z22get_sub_group_local_idv() #2

declare spir_func void @__sub_group_commit_write_pipe(target("spirv.Pipe", 1), target("spirv.ReserveId"), i32, i32)

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test_pipe_subgroup_read_uint(target("spirv.Pipe", 0) %in_pipe, ptr addrspace(1) noundef align 4 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
entry:
  %in_pipe.addr = alloca target("spirv.Pipe", 0), align 8
  %dst.addr = alloca ptr addrspace(1), align 8
  %gid = alloca i32, align 4
  %res_id = alloca target("spirv.ReserveId"), align 8
  store target("spirv.Pipe", 0) %in_pipe, ptr %in_pipe.addr, align 8
  store ptr addrspace(1) %dst, ptr %dst.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %gid, align 4
  %0 = load target("spirv.Pipe", 0), ptr %in_pipe.addr, align 8
  %call1 = call spir_func i32 @_Z18get_sub_group_sizev() #4
  %1 = call spir_func target("spirv.ReserveId") @__sub_group_reserve_read_pipe(target("spirv.Pipe", 0) %0, i32 %call1, i32 4, i32 4)
  store target("spirv.ReserveId") %1, ptr %res_id, align 8
  %2 = load target("spirv.ReserveId"), ptr %res_id, align 8
  %call2 = call spir_func zeroext i1 @_Z19is_valid_reserve_id13ocl_reserveid(target("spirv.ReserveId") %2) #4
  br i1 %call2, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %3 = load target("spirv.Pipe", 0), ptr %in_pipe.addr, align 8
  %4 = load target("spirv.ReserveId"), ptr %res_id, align 8
  %call3 = call spir_func i32 @_Z22get_sub_group_local_idv() #4
  %5 = load ptr addrspace(1), ptr %dst.addr, align 8
  %6 = load i32, ptr %gid, align 4
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %idxprom
  %7 = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  %8 = call spir_func i32 @__read_pipe_4(target("spirv.Pipe", 0) %3, target("spirv.ReserveId") %4, i32 %call3, ptr addrspace(4) %7, i32 4, i32 4)
  %9 = load target("spirv.Pipe", 0), ptr %in_pipe.addr, align 8
  %10 = load target("spirv.ReserveId"), ptr %res_id, align 8
  call spir_func void @__sub_group_commit_read_pipe(target("spirv.Pipe", 0) %9, target("spirv.ReserveId") %10, i32 4, i32 4)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare spir_func target("spirv.ReserveId") @__sub_group_reserve_read_pipe(target("spirv.Pipe", 0), i32, i32, i32)

declare spir_func void @__sub_group_commit_read_pipe(target("spirv.Pipe", 0), target("spirv.ReserveId"), i32, i32)

attributes #0 = { convergent noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind willreturn memory(none) }
attributes #4 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"Ubuntu clang version 19.1.7 (++20250114103320+cd708029e0b2-1~exp1~20250114103432.75)"}
!3 = !{i32 1, i32 1}
!4 = !{!"none", !"write_only"}
!5 = !{!"uint*", !"uint"}
!6 = !{!"", !"pipe"}
!7 = !{!"read_only", !"none"}
!8 = !{!"uint", !"uint*"}
!9 = !{!"pipe", !""}
!10 = !{!"int*", !"int"}
!11 = !{i32 1, i32 1, i32 1}
!12 = !{!"write_only", !"none", !"none"}
!13 = !{!"int", !"int*", !"int*"}
!14 = !{!"pipe", !"", !""}
!15 = !{!"int", !"int*"}
!16 = !{!"char*", !"char"}
!17 = !{!"char", !"char*"}
