// RUN: mlir-opt %s -canonicalize | mlir-opt | FileCheck %s

module attributes {mpi.dlti = #dlti.map<"MPI:comm_world_size" = 12, "MPI:comm_world_rank" = 5> } {
  // CHECK-LABEL: func.func @mpi_test
  func.func @mpi_test(%ref : memref<100xf32>) -> (i32, i32) {
    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[s:%.*]] = arith.constant 12 : i32
    %sz = mpi.comm_size(%comm) : i32
    // CHECK: [[r:%.*]] = arith.constant 5 : i32
    %rk = mpi.comm_rank(%comm) : i32
    // CHECK: return [[s]], [[r]] : i32, i32
    return %sz, %rk : i32, i32
  }
}
