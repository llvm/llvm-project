// Pipe built-ins are mangled accordingly to SPIR2.0/C++ ABI.

// RUN: %clang_cc1 -x cl -cl-std=CL2.0 -triple spir64-unknonw-unknown -emit-llvm-bc -finclude-default-header -Dcl_khr_subgroups %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.bc
// RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV-DAG: TypePipe [[ROPipeTy:[0-9]+]] 0
// CHECK-SPIRV-DAG: TypePipe [[WOPipeTy:[0-9]+]] 1

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// CHECK-LLVM-LABEL: @test_pipe_convenience_write_uint
__kernel void test_pipe_convenience_write_uint(__global uint *src, __write_only pipe uint out_pipe)
{
  // CHECK-SPIRV: 5 Function
  // CHECK-SPIRV-NEXT:  FunctionParameter
  // CHECK-SPIRV-NEXT:  FunctionParameter [[WOPipeTy]] [[PipeArgID:[0-9]+]]
  // CHECK-LLVM: call{{.*}}@__write_pipe_2
  // CHECK-SPIRV: WritePipe {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-SPIRV-LABEL: 1 FunctionEnd

  int gid = get_global_id(0);
  write_pipe(out_pipe, &src[gid]);
}

// CHECK-LLVM-LABEL: @test_pipe_convenience_read_uint
// CHECK-SPIRV-LABEL: 5 Function
__kernel void test_pipe_convenience_read_uint(__read_only pipe uint in_pipe, __global uint *dst)
{
  // CHECK-SPIRV-NEXT:  FunctionParameter [[ROPipeTy]] [[PipeArgID:[0-9]+]]
  // CHECK-LLVM: call{{.*}}@__read_pipe_2
  // CHECK-SPIRV: ReadPipe {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-SPIRV-LABEL: 1 FunctionEnd

  int gid = get_global_id(0);
  read_pipe(in_pipe, &dst[gid]);
}

// CHECK-LLVM-LABEL: @test_pipe_write
// CHECK-SPIRV-LABEL: 5 Function
__kernel void test_pipe_write(__global int *src, __write_only pipe int out_pipe)
{
  // CHECK-SPIRV-NEXT:  FunctionParameter
  // CHECK-SPIRV-NEXT:  FunctionParameter [[WOPipeTy]] [[PipeArgID:[0-9]+]]
  // CHECK-LLVM: @__reserve_write_pipe
  // CHECK-SPIRV: ReserveWritePipePackets {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-LLVM: call{{.*}}@__write_pipe_4
  // CHECK-SPIRV: ReservedWritePipe {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-LLVM: call{{.*}}@__commit_write_pipe
  // CHECK-SPIRV: CommitWritePipe [[PipeArgID]] {{[0-9]+}} {{[0-9]+}}
  // CHECK-SPIRV-LABEL: 1 FunctionEnd

  int gid = get_global_id(0);
  reserve_id_t res_id;
  res_id = reserve_write_pipe(out_pipe, 1);
  if(is_valid_reserve_id(res_id))
  {
    write_pipe(out_pipe, res_id, 0, &src[gid]);
    commit_write_pipe(out_pipe, res_id);
  }
}

// CHECK-LLVM-LABEL: @test_pipe_query_functions
// CHECK-SPIRV-LABEL: 5 Function
__kernel void test_pipe_query_functions(__write_only pipe int out_pipe, __global int *num_packets, __global int *max_packets)
{
  // CHECK-SPIRV-NEXT:  FunctionParameter [[WOPipeTy]] [[PipeArgID:[0-9]+]]
  // CHECK-LLVM: call{{.*}}@__get_pipe_max_packets_wo
  // CHECK-SPIRV: GetMaxPipePackets {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}}
  // CHECK-LLVM: call{{.*}}@__get_pipe_num_packets
  // CHECK-SPIRV: GetNumPipePackets {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}}
  // CHECK-SPIRV-LABEL: 1 FunctionEnd

  *max_packets = get_pipe_max_packets(out_pipe);
  *num_packets = get_pipe_num_packets(out_pipe);
}

// CHECK-LLVM-LABEL: @test_pipe_read
// CHECK-SPIRV-LABEL: 5 Function
__kernel void test_pipe_read(__read_only pipe int in_pipe, __global int *dst)
{
  // CHECK-SPIRV-NEXT:  FunctionParameter [[ROPipeTy]] [[PipeArgID:[0-9]+]]
  // CHECK-LLVM: call{{.*}}@__reserve_read_pipe
  // CHECK-SPIRV: ReserveReadPipePackets {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-LLVM: call{{.*}}@__read_pipe_4
  // CHECK-SPIRV: ReservedReadPipe {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-LLVM: call{{.*}}@__commit_read_pipe
  // CHECK-SPIRV: CommitReadPipe [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-SPIRV-LABEL: 1 FunctionEnd

  int gid = get_global_id(0);
  reserve_id_t res_id;
  res_id = reserve_read_pipe(in_pipe, 1);
  if(is_valid_reserve_id(res_id))
  {
    read_pipe(in_pipe, res_id, 0, &dst[gid]);
    commit_read_pipe(in_pipe, res_id);
  }
}

// CHECK-LLVM-LABEL: @test_pipe_workgroup_write_char
// CHECK-SPIRV-LABEL: 5 Function
__kernel void test_pipe_workgroup_write_char(__global char *src, __write_only pipe char out_pipe)
{
  // CHECK-SPIRV-NEXT:  FunctionParameter
  // CHECK-SPIRV-NEXT:  FunctionParameter [[WOPipeTy]] [[PipeArgID:[0-9]+]]
  // CHECK-LLVM: call{{.*}}@__work_group_reserve_write_pipe
  // CHECK-SPIRV: GroupReserveWritePipePackets {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-LLVM: call{{.*}}@__work_group_commit_write_pipe
  // CHECK-SPIRV: GroupCommitWritePipe {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-SPIRV-LABEL: 1 FunctionEnd

  int gid = get_global_id(0);
  __local reserve_id_t res_id;

  res_id = work_group_reserve_write_pipe(out_pipe, get_local_size(0));
  if(is_valid_reserve_id(res_id))
  {
    write_pipe(out_pipe, res_id, get_local_id(0), &src[gid]);
    work_group_commit_write_pipe(out_pipe, res_id);
  }
}

// CHECK-LLVM-LABEL: @test_pipe_workgroup_read_char
// CHECK-SPIRV-LABEL: 5 Function
__kernel void test_pipe_workgroup_read_char(__read_only pipe char in_pipe, __global char *dst)
{
  // CHECK-SPIRV-NEXT:  FunctionParameter [[ROPipeTy]] [[PipeArgID:[0-9]+]]
  // CHECK-LLVM: call{{.*}}@__work_group_reserve_read_pipe
  // CHECK-SPIRV: GroupReserveReadPipePackets {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-LLVM: call{{.*}}@__work_group_commit_read_pipe
  // CHECK-SPIRV: GroupCommitReadPipe {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-SPIRV-LABEL: 1 FunctionEnd

  int gid = get_global_id(0);
  __local reserve_id_t res_id;

  res_id = work_group_reserve_read_pipe(in_pipe, get_local_size(0));
  if(is_valid_reserve_id(res_id))
  {
    read_pipe(in_pipe, res_id, get_local_id(0), &dst[gid]);
    work_group_commit_read_pipe(in_pipe, res_id);
  }
}

// CHECK-LLVM-LABEL: @test_pipe_subgroup_write_uint
// CHECK-SPIRV-LABEL: 5 Function
__kernel void test_pipe_subgroup_write_uint(__global uint *src, __write_only pipe uint out_pipe)
{
  // CHECK-SPIRV-NEXT:  FunctionParameter
  // CHECK-SPIRV-NEXT:  FunctionParameter [[WOPipeTy]] [[PipeArgID:[0-9]+]]
  // CHECK-LLVM: call{{.*}}@__sub_group_reserve_write_pipe
  // CHECK-SPIRV: GroupReserveWritePipePackets {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-LLVM: call{{.*}}@__sub_group_commit_write_pipe
  // CHECK-SPIRV: GroupCommitWritePipe {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}

  int gid = get_global_id(0);
  reserve_id_t res_id;

  res_id = sub_group_reserve_write_pipe(out_pipe, get_sub_group_size());
  if(is_valid_reserve_id(res_id))
  {
    write_pipe(out_pipe, res_id, get_sub_group_local_id(), &src[gid]);
    sub_group_commit_write_pipe(out_pipe, res_id);
  }
}

// CHECK-LLVM-LABEL: @test_pipe_subgroup_read_uint
// CHECK-SPIRV-LABEL: 5 Function
__kernel void test_pipe_subgroup_read_uint(__read_only pipe uint in_pipe, __global uint *dst)
{
  // CHECK-SPIRV-NEXT:  FunctionParameter [[ROPipeTy]] [[PipeArgID:[0-9]+]]
  // CHECK-LLVM: call{{.*}}@__sub_group_reserve_read_pipe
  // CHECK-SPIRV: GroupReserveReadPipePackets {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-LLVM: call{{.*}}@__sub_group_commit_read_pipe
  // CHECK-SPIRV: GroupCommitReadPipe {{[0-9]+}} [[PipeArgID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
  // CHECK-SPIRV-LABEL: 1 FunctionEnd

  int gid = get_global_id(0);
  reserve_id_t res_id;

  res_id = sub_group_reserve_read_pipe(in_pipe, get_sub_group_size());
  if(is_valid_reserve_id(res_id))
  {
    read_pipe(in_pipe, res_id, get_sub_group_local_id(), &dst[gid]);
    sub_group_commit_read_pipe(in_pipe, res_id);
  }
}
