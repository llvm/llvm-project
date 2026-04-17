// RUN: mlir-opt -split-input-file -convert-to-llvm %s | FileCheck %s

// COM: Test MPICH ABI
// CHECK-LABEL: module attributes {dlti.map = #dlti.map<"MPI:Implementation" = "MPICH">} {
// CHECK-DAG: llvm.func @MPI_Finalize() -> i32
// CHECK-DAG: llvm.func @MPI_Reduce_scatter_block(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK-DAG: llvm.func @MPI_Allreduce(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK-DAG: llvm.func @MPI_Comm_size(i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Allgather(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32) -> i32
// CHECK-DAG: llvm.func @MPI_Comm_split(i32, i32, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Recv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Send(!llvm.ptr, i32, i32, i32, i32, i32) -> i32
// CHECK-DAG: llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
module attributes {dlti.map = #dlti.map<"MPI:Implementation" = "MPICH">} {

  // CHECK-LABEL: llvm.func @test_init_finalize_mpich
  func.func @test_init_finalize_mpich() {
    // CHECK: [[v0:%.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.call @MPI_Init([[v0]], [[v0]]) : (!llvm.ptr, !llvm.ptr) -> i32
    %0 = mpi.init : !mpi.retval
    // CHECK: llvm.call @MPI_Finalize() : () -> i32
    %1 = mpi.finalize : !mpi.retval
    return
  }

  // CHECK-LABEL: llvm.func @test_comm_rank_mpich
  func.func @test_comm_rank_mpich() {
    // CHECK: [[v0:%.*]] = llvm.mlir.constant(1140850688 : i64) : i64
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v1:%.*]] = llvm.trunc [[v0]] : i64 to i32
    // CHECK: [[v2:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[v3:%.*]] = llvm.alloca [[v2]] x i32 : (i32) -> !llvm.ptr
    // CHECK: llvm.call @MPI_Comm_rank([[v1]], [[v3]]) : (i32, !llvm.ptr) -> i32
    %retval, %rank = mpi.comm_rank(%comm) : !mpi.retval, i32
    return
  }

  // CHECK-LABEL: llvm.func @test_send_mpich
  func.func @test_send_mpich(%arg0: memref<100xf32>) {
    // CHECK: [[v0:%.*]] = llvm.insertvalue {{.*}}[4, 0]
    // CHECK: [[v1:%.*]] = llvm.mlir.constant(1140850688 : i64) : i64
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: llvm.call @MPI_Comm_rank
    // CHECK: [[v2:%.*]] = llvm.load {{.*}} : !llvm.ptr -> i32
    %retval, %rank = mpi.comm_rank(%comm) : !mpi.retval, i32
    // COM: Test send without retval
    // CHECK: [[v3:%.*]] = llvm.extractvalue [[v0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v4:%.*]] = llvm.extractvalue [[v0]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v5:%.*]] = llvm.getelementptr [[v3]][[[v4]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v6:%.*]] = llvm.extractvalue [[v0]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v7:%.*]] = llvm.trunc [[v6]] : i64 to i32
    // CHECK: [[v8:%.*]] = llvm.mul [[v7]]
    // CHECK: [[v9:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[v10:%.*]] = llvm.trunc [[v1]] : i64 to i32
    // CHECK: = llvm.call @MPI_Send([[v5]], [[v8]], [[v9]], [[v2]], [[v2]], [[v10]]) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
    mpi.send(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32
    // COM: Test send with retval
    // CHECK: = llvm.call @MPI_Send({{.*}}) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
    %1 = mpi.send(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval
    return
  }

  // CHECK-LABEL: llvm.func @test_recv_mpich
  func.func @test_recv_mpich(%arg0: memref<100xf32>) {
    // CHECK: [[v0:%.*]] = llvm.insertvalue {{.*}}[4, 0]
    // CHECK: [[v1:%.*]] = llvm.mlir.constant(1140850688 : i64) : i64
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: llvm.call @MPI_Comm_rank
    // CHECK: [[v2:%.*]] = llvm.load {{.*}} : !llvm.ptr -> i32
    %retval, %rank = mpi.comm_rank(%comm) : !mpi.retval, i32
    // COM: Test recv without retval
    // CHECK: [[v3:%.*]] = llvm.extractvalue [[v0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v4:%.*]] = llvm.extractvalue [[v0]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v5:%.*]] = llvm.getelementptr [[v3]][[[v4]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v6:%.*]] = llvm.extractvalue [[v0]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v7:%.*]] = llvm.trunc [[v6]] : i64 to i32
    // CHECK: [[v8:%.*]] = llvm.mul [[v7]]
    // CHECK: [[v9:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[v10:%.*]] = llvm.trunc [[v1]] : i64 to i32
    // CHECK: [[v11:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[v12:%.*]] = llvm.inttoptr [[v11]] : i64 to !llvm.ptr
    // CHECK: = llvm.call @MPI_Recv([[v5]], [[v8]], [[v9]], [[v2]], [[v2]], [[v10]], [[v12]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
    mpi.recv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32
    // COM: Test recv with retval
    // CHECK: = llvm.call @MPI_Recv({{.*}}) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
    %2 = mpi.recv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval
    return
  }

  // CHECK-LABEL: llvm.func @test_comm_split_mpich
  func.func @test_comm_split_mpich() {
    // CHECK: [[v0:%.*]] = llvm.mlir.constant(1140850688 : i64) : i64
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v1:%.*]] = llvm.mlir.constant(10 : i32) : i32
    %color = arith.constant 10 : i32
    // CHECK: [[v2:%.*]] = llvm.mlir.constant(22 : i32) : i32
    %key = arith.constant 22 : i32
    // CHECK: [[v3:%.*]] = llvm.trunc [[v0]] : i64 to i32
    // CHECK: [[v4:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[v5:%.*]] = llvm.alloca [[v4]] x i32 : (i32) -> !llvm.ptr
    // CHECK: llvm.call @MPI_Comm_split([[v3]], [[v1]], [[v2]], [[v5]]) : (i32, i32, i32, !llvm.ptr) -> i32
    // CHECK: llvm.load [[v5]] : !llvm.ptr -> i32
    %split = mpi.comm_split(%comm, %color, %key) : !mpi.comm
    return
  }

  // CHECK-LABEL: llvm.func @test_allgather_mpich
  func.func @test_allgather_mpich(%arg0: memref<100xf32>) {
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: llvm.call @MPI_Comm_size
    // CHECK: llvm.call @MPI_Allgather({{.*}}) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32) -> i32
    %err = mpi.allgather(%arg0, %arg0, %comm) : memref<100xf32>, memref<100xf32> -> !mpi.retval
    return
  }

  // CHECK-LABEL: llvm.func @test_allreduce_mpich
  func.func @test_allreduce_mpich(%arg0: memref<100xf32>) {
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v0:%.*]] = llvm.insertvalue {{.*}}[4, 0]
    // CHECK: [[v1:%.*]] = llvm.mlir.constant(1140850688 : i64) : i64
    // CHECK: [[v2:%.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v3:%.*]] = llvm.mul
    // CHECK: [[v4:%.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v5:%.*]] = llvm.mlir.constant(-1 : i64) : i64
    // CHECK: [[v6:%.*]] = llvm.inttoptr [[v5]] : i64 to !llvm.ptr
    // CHECK: [[v7:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[v8:%.*]] = llvm.mlir.constant(1476395011 : i32) : i32
    // CHECK: [[v9:%.*]] = llvm.trunc [[v1]] : i64 to i32
    // CHECK: llvm.call @MPI_Allreduce([[v6]], [[v4]], [[v3]], [[v7]], [[v8]], [[v9]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
    mpi.allreduce(%arg0, %arg0, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>
    return
  }

  // CHECK-LABEL: llvm.func @test_reduce_scatter_block_mpich
  func.func @test_reduce_scatter_block_mpich(%arg0: memref<100xf32>) {
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v0:%.*]] = llvm.insertvalue {{.*}}[4, 0]
    // CHECK: llvm.mul
    // CHECK: [[v1:%.*]] = llvm.mlir.constant(1 : index) : i32
    // CHECK: [[v2:%.*]] = llvm.extractvalue [[v0]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v3:%.*]] = llvm.trunc [[v2]] : i64 to i32
    // CHECK: [[v4:%.*]] = llvm.mul [[v3]], [[v1]] : i32
    // CHECK: [[v5:%.*]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr
    // CHECK: llvm.cond_br {{.*}}, ^[[bb1:.*]], ^{{.*}}
    // CHECK: ^[[bb1]]:
    // CHECK: llvm.call @MPI_Reduce_scatter_block([[v5]], {{.*}}, [[v4]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
    mpi.reduce_scatter_block(%arg0, %arg0, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>
    return
  }
}

// -----

// COM: Test OpenMPI ABI
// CHECK-LABEL: module attributes {dlti.map = #dlti.map<"MPI:Implementation" = "OpenMPI">} {
// CHECK-DAG: llvm.func @MPI_Finalize() -> i32
// CHECK-DAG: llvm.func @MPI_Comm_split(!llvm.ptr, i32, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Reduce_scatter_block(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Allreduce(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global external @ompi_mpi_sum() {addr_space = 0 : i32} : !llvm.struct<"ompi_predefined_op_t", opaque>
// CHECK-DAG: llvm.func @MPI_Comm_size(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Allgather(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Recv(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Send(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global external @ompi_mpi_float() {addr_space = 0 : i32} : !llvm.struct<"ompi_predefined_datatype_t", opaque>
// CHECK-DAG: llvm.func @MPI_Comm_rank(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global external @ompi_mpi_comm_world() {addr_space = 0 : i32} : !llvm.struct<"ompi_communicator_t", opaque>
// CHECK-DAG: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
module attributes { dlti.map = #dlti.map<"MPI:Implementation" = "OpenMPI"> } {

  // CHECK-LABEL: llvm.func @test_init_finalize_openmpi
  func.func @test_init_finalize_openmpi() {
    // CHECK: [[v0:%.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.call @MPI_Init([[v0]], [[v0]]) : (!llvm.ptr, !llvm.ptr) -> i32
    %0 = mpi.init : !mpi.retval
    // CHECK: llvm.call @MPI_Finalize() : () -> i32
    %1 = mpi.finalize : !mpi.retval
    return
  }

  // CHECK-LABEL: llvm.func @test_comm_rank_openmpi
  func.func @test_comm_rank_openmpi() {
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v0:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[v1:%.*]] = llvm.ptrtoint [[v0]] : !llvm.ptr to i64
    // CHECK: [[v2:%.*]] = llvm.inttoptr [[v1]] : i64 to !llvm.ptr
    // CHECK: [[v3:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[v4:%.*]] = llvm.alloca [[v3]] x i32 : (i32) -> !llvm.ptr
    // CHECK: llvm.call @MPI_Comm_rank([[v2]], [[v4]]) : (!llvm.ptr, !llvm.ptr) -> i32
    %retval, %rank = mpi.comm_rank(%comm) : !mpi.retval, i32
    return
  }

  // CHECK-LABEL: llvm.func @test_send_openmpi
  func.func @test_send_openmpi(%arg0: memref<100xf32>) {
    // CHECK: [[v0:%.*]] = llvm.insertvalue {{.*}}[4, 0]
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v1:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[v2:%.*]] = llvm.ptrtoint [[v1]] : !llvm.ptr to i64
    // CHECK: llvm.call @MPI_Comm_rank
    // CHECK: [[v3:%.*]] = llvm.load {{.*}} : !llvm.ptr -> i32
    %retval, %rank = mpi.comm_rank(%comm) : !mpi.retval, i32
    // COM: Test send without retval
    // CHECK: [[v4:%.*]] = llvm.extractvalue [[v0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v5:%.*]] = llvm.extractvalue [[v0]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v6:%.*]] = llvm.getelementptr [[v4]][[[v5]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v7:%.*]] = llvm.extractvalue [[v0]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v8:%.*]] = llvm.trunc [[v7]] : i64 to i32
    // CHECK: [[v9:%.*]] = llvm.mul [[v8]]
    // CHECK: [[v10:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v11:%.*]] = llvm.inttoptr [[v2]] : i64 to !llvm.ptr
    // CHECK: = llvm.call @MPI_Send([[v6]], [[v9]], [[v10]], [[v3]], [[v3]], [[v11]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
    mpi.send(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32
    // COM: Test send with retval
    // CHECK: = llvm.call @MPI_Send({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
    %1 = mpi.send(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval
    return
  }

  // CHECK-LABEL: llvm.func @test_recv_openmpi
  func.func @test_recv_openmpi(%arg0: memref<100xf32>) {
    // CHECK: [[v0:%.*]] = llvm.insertvalue {{.*}}[4, 0]
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v1:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[v2:%.*]] = llvm.ptrtoint [[v1]] : !llvm.ptr to i64
    // CHECK: llvm.call @MPI_Comm_rank
    // CHECK: [[v3:%.*]] = llvm.load {{.*}} : !llvm.ptr -> i32
    %retval, %rank = mpi.comm_rank(%comm) : !mpi.retval, i32
    // COM: Test recv without retval
    // CHECK: [[v4:%.*]] = llvm.extractvalue [[v0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v5:%.*]] = llvm.extractvalue [[v0]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v6:%.*]] = llvm.getelementptr [[v4]][[[v5]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v7:%.*]] = llvm.extractvalue [[v0]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v8:%.*]] = llvm.trunc [[v7]] : i64 to i32
    // CHECK: [[v9:%.*]] = llvm.mul [[v8]]
    // CHECK: [[v10:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v11:%.*]] = llvm.inttoptr [[v2]] : i64 to !llvm.ptr
    // CHECK: [[v12:%.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[v13:%.*]] = llvm.inttoptr [[v12]] : i64 to !llvm.ptr
    // CHECK: = llvm.call @MPI_Recv([[v6]], [[v9]], [[v10]], [[v3]], [[v3]], [[v11]], [[v13]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    mpi.recv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32
    // COM: Test recv with retval
    // CHECK: = llvm.call @MPI_Recv({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    %2 = mpi.recv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval
    return
  }

  // CHECK-LABEL: llvm.func @test_comm_split_openmpi
  func.func @test_comm_split_openmpi() {
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v0:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[v1:%.*]] = llvm.ptrtoint [[v0]] : !llvm.ptr to i64
    // CHECK: [[v2:%.*]] = llvm.mlir.constant(10 : i32) : i32
    %color = arith.constant 10 : i32
    // CHECK: [[v3:%.*]] = llvm.mlir.constant(22 : i32) : i32
    %key = arith.constant 22 : i32
    // CHECK: [[v4:%.*]] = llvm.inttoptr [[v1]] : i64 to !llvm.ptr
    // CHECK: [[v5:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[v6:%.*]] = llvm.alloca [[v5]] x !llvm.ptr : (i32) -> !llvm.ptr
    // CHECK: llvm.call @MPI_Comm_split([[v4]], [[v2]], [[v3]], [[v6]]) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i32
    // CHECK: llvm.load [[v6]] : !llvm.ptr -> i32
    %split = mpi.comm_split(%comm, %color, %key) : !mpi.comm
    return
  }

  // CHECK-LABEL: llvm.func @test_allgather_openmpi
  func.func @test_allgather_openmpi(%arg0: memref<100xf32>) {
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: llvm.call @MPI_Comm_size({{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.udiv {{.*}} : i32
    // CHECK: llvm.call @MPI_Allgather({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
    mpi.allgather(%arg0, %arg0, %comm) : memref<100xf32>, memref<100xf32>
    return
  }

  // CHECK-LABEL: llvm.func @test_allreduce_openmpi
  func.func @test_allreduce_openmpi(%arg0: memref<100xf32>) {
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v0:%.*]] = llvm.insertvalue {{.*}}[4, 0]
    // CHECK: [[v1:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[v2:%.*]] = llvm.ptrtoint [[v1]] : !llvm.ptr to i64
    // CHECK: [[v3:%.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v4:%.*]] = llvm.mul
    // CHECK: [[v5:%.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v6:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[v7:%.*]] = llvm.inttoptr [[v6]] : i64 to !llvm.ptr
    // CHECK: [[v8:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v9:%.*]] = llvm.mlir.addressof @ompi_mpi_sum : !llvm.ptr
    // CHECK: [[v10:%.*]] = llvm.inttoptr [[v2]] : i64 to !llvm.ptr
    // CHECK: llvm.call @MPI_Allreduce([[v7]], [[v5]], [[v4]], [[v8]], [[v9]], [[v10]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    mpi.allreduce(%arg0, %arg0, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>
    return
  }

  // CHECK-LABEL: llvm.func @test_reduce_scatter_block_openmpi
  func.func @test_reduce_scatter_block_openmpi(%arg0: memref<100xf32>) {
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v0:%.*]] = llvm.insertvalue {{.*}}[4, 0]
    // CHECK: llvm.mul
    // CHECK: [[v1:%.*]] = llvm.mlir.constant(1 : index) : i32
    // CHECK: [[v2:%.*]] = llvm.extractvalue [[v0]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v3:%.*]] = llvm.trunc [[v2]] : i64 to i32
    // CHECK: [[v4:%.*]] = llvm.mul [[v3]], [[v1]] : i32
    // CHECK: [[v5:%.*]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr
    // CHECK: llvm.cond_br {{.*}}, ^[[bb1:.*]], ^{{.*}}
    // CHECK: ^[[bb1]]:
    // CHECK: llvm.call @MPI_Reduce_scatter_block([[v5]], {{.*}}, [[v4]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    mpi.reduce_scatter_block(%arg0, %arg0, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>
    return
  }
}

// -----

module attributes {mpi.dlti = #dlti.map<"MPI:Implementation" = "MPICH", "MPI:comm_world_size" = 4, "MPI:comm_world_rank" = 1> } {
  // CHECK-LABEL: llvm.func @test_fold
  func.func @test_fold(%arg0: memref<100xf32>) {
    // CHECK: [[v0:%.*]] = llvm.mlir.constant(1140850688 : i64) : i64
    %comm = mpi.comm_world : !mpi.comm

    // CHECK-NOT: llvm.call @MPI_Comm_size
    // CHECK: llvm.call @MPI_Allgather({{.*}}) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32) -> i32
    %err3 = mpi.allgather(%arg0, %arg0, %comm) : memref<100xf32>, memref<100xf32> -> !mpi.retval
    return
  }
}
