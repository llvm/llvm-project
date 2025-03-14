// RUN: mlir-opt -split-input-file -convert-to-llvm %s | FileCheck %s

// COM: Test MPICH ABI
// CHECK: module attributes {mpi.dlti = #dlti.map<"MPI:Implementation" = "MPICH">} {
// CHECK: llvm.func @MPI_Finalize() -> i32
// CHECK: llvm.func @MPI_Recv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK: llvm.func @MPI_Send(!llvm.ptr, i32, i32, i32, i32, i32) -> i32
// CHECK: llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
// CHECK: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
module attributes { mpi.dlti = #dlti.map<"MPI:Implementation" = "MPICH"> } {

  // CHECK: llvm.func @mpi_test_mpich([[varg0:%.+]]: !llvm.ptr, [[varg1:%.+]]: !llvm.ptr, [[varg2:%.+]]: i64, [[varg3:%.+]]: i64, [[varg4:%.+]]: i64) {
  func.func @mpi_test_mpich(%arg0: memref<100xf32>) {

    // CHECK: [[v0:%.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v1:%.*]] = llvm.insertvalue [[varg0]], [[v0]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v2:%.*]] = llvm.insertvalue [[varg1]], [[v1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v3:%.*]] = llvm.insertvalue [[varg2]], [[v2]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v4:%.*]] = llvm.insertvalue [[varg3]], [[v3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v5:%.*]] = llvm.insertvalue [[varg4]], [[v4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v6:%.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: [[v7:%.*]] = llvm.call @MPI_Init([[v6]], [[v6]]) : (!llvm.ptr, !llvm.ptr) -> i32
    %0 = mpi.init : !mpi.retval

    // CHECK: [[v8:%.*]] = llvm.mlir.constant(1140850688 : i32) : i32
    // CHECK: [[v9:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[v10:%.*]] = llvm.alloca [[v9]] x i32 : (i32) -> !llvm.ptr
    // CHECK: [[v11:%.*]] = llvm.call @MPI_Comm_rank([[v8]], [[v10]]) : (i32, !llvm.ptr) -> i32
    %retval, %rank = mpi.comm_rank : !mpi.retval, i32

    // CHECK: [[v12:%.*]] = llvm.load [[v10]] : !llvm.ptr -> i32
    // CHECK: [[v13:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v14:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v15:%.*]] = llvm.getelementptr [[v13]][[[v14]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v16:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v17:%.*]] = llvm.trunc [[v16]] : i64 to i32
    // CHECK: [[v18:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[v19:%.*]] = llvm.mlir.constant(1140850688 : i32) : i32
    // CHECK: [[v20:%.*]] = llvm.call @MPI_Send([[v15]], [[v17]], [[v18]], [[v12]], [[v12]], [[v19]]) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
    mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK: [[v21:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v22:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v23:%.*]] = llvm.getelementptr [[v21]][[[v22]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v24:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v25:%.*]] = llvm.trunc [[v24]] : i64 to i32
    // CHECK: [[v26:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[v27:%.*]] = llvm.mlir.constant(1140850688 : i32) : i32
    // CHECK: [[v28:%.*]] = llvm.call @MPI_Send([[v23]], [[v25]], [[v26]], [[v12]], [[v12]], [[v27]]) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
    %1 = mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: [[v29:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v30:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v31:%.*]] = llvm.getelementptr [[v29]][[[v30]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v32:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v33:%.*]] = llvm.trunc [[v32]] : i64 to i32
    // CHECK: [[v34:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[v35:%.*]] = llvm.mlir.constant(1140850688 : i32) : i32
    // CHECK: [[v36:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[v37:%.*]] = llvm.inttoptr [[v36]] : i64 to !llvm.ptr
    // CHECK: [[v38:%.*]] = llvm.call @MPI_Recv([[v31]], [[v33]], [[v34]], [[v12]], [[v12]], [[v35]], [[v37]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
    mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK: [[v39:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v40:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v41:%.*]] = llvm.getelementptr [[v39]][[[v40]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v42:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v43:%.*]] = llvm.trunc [[v42]] : i64 to i32
    // CHECK: [[v44:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[v45:%.*]] = llvm.mlir.constant(1140850688 : i32) : i32
    // CHECK: [[v46:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[v47:%.*]] = llvm.inttoptr [[v46]] : i64 to !llvm.ptr
    // CHECK: [[v48:%.*]] = llvm.call @MPI_Recv([[v41]], [[v43]], [[v44]], [[v12]], [[v12]], [[v45]], [[v47]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
    %2 = mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: [[v49:%.*]] = llvm.call @MPI_Finalize() : () -> i32
    %3 = mpi.finalize : !mpi.retval

    return
  }
}

// -----

// COM: Test OpenMPI ABI
// CHECK: module attributes {mpi.dlti = #dlti.map<"MPI:Implementation" = "OpenMPI">} {
// CHECK: llvm.func @MPI_Finalize() -> i32
// CHECK: llvm.func @MPI_Recv(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.func @MPI_Send(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
// CHECK: llvm.mlir.global external @ompi_mpi_float() {addr_space = 0 : i32} : !llvm.struct<"ompi_predefined_datatype_t", opaque>
// CHECK: llvm.func @MPI_Comm_rank(!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.mlir.global external @ompi_mpi_comm_world() {addr_space = 0 : i32} : !llvm.struct<"ompi_communicator_t", opaque>
// CHECK: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
module attributes { mpi.dlti = #dlti.map<"MPI:Implementation" = "OpenMPI"> } {

  // CHECK: llvm.func @mpi_test_openmpi([[varg0:%.+]]: !llvm.ptr, [[varg1:%.+]]: !llvm.ptr, [[varg2:%.+]]: i64, [[varg3:%.+]]: i64, [[varg4:%.+]]: i64) {
  func.func @mpi_test_openmpi(%arg0: memref<100xf32>) {

    // CHECK: [[v0:%.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v1:%.*]] = llvm.insertvalue [[varg0]], [[v0]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v2:%.*]] = llvm.insertvalue [[varg1]], [[v1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v3:%.*]] = llvm.insertvalue [[varg2]], [[v2]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v4:%.*]] = llvm.insertvalue [[varg3]], [[v3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v5:%.*]] = llvm.insertvalue [[varg4]], [[v4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v6:%.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: [[v7:%.*]] = llvm.call @MPI_Init([[v6]], [[v6]]) : (!llvm.ptr, !llvm.ptr) -> i32
    %0 = mpi.init : !mpi.retval

    // CHECK: [[v8:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[v9:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[v10:%.*]] = llvm.alloca [[v9]] x i32 : (i32) -> !llvm.ptr
    // CHECK: [[v11:%.*]] = llvm.call @MPI_Comm_rank([[v8]], [[v10]]) : (!llvm.ptr, !llvm.ptr) -> i32
    %retval, %rank = mpi.comm_rank : !mpi.retval, i32

    // CHECK: [[v12:%.*]] = llvm.load [[v10]] : !llvm.ptr -> i32
    // CHECK: [[v13:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v14:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v15:%.*]] = llvm.getelementptr [[v13]][[[v14]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v16:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v17:%.*]] = llvm.trunc [[v16]] : i64 to i32
    // CHECK: [[v18:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v19:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[v20:%.*]] = llvm.call @MPI_Send([[v15]], [[v17]], [[v18]], [[v12]], [[v12]], [[v19]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
    mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK: [[v21:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v22:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v23:%.*]] = llvm.getelementptr [[v21]][[[v22]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v24:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v25:%.*]] = llvm.trunc [[v24]] : i64 to i32
    // CHECK: [[v26:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v27:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[v28:%.*]] = llvm.call @MPI_Send([[v23]], [[v25]], [[v26]], [[v12]], [[v12]], [[v27]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
    %1 = mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: [[v29:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v30:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v31:%.*]] = llvm.getelementptr [[v29]][[[v30]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v32:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v33:%.*]] = llvm.trunc [[v32]] : i64 to i32
    // CHECK: [[v34:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v35:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[v36:%.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[v37:%.*]] = llvm.inttoptr [[v36]] : i64 to !llvm.ptr
    // CHECK: [[v38:%.*]] = llvm.call @MPI_Recv([[v31]], [[v33]], [[v34]], [[v12]], [[v12]], [[v35]], [[v37]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK: [[v39:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v40:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v41:%.*]] = llvm.getelementptr [[v39]][[[v40]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v42:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v43:%.*]] = llvm.trunc [[v42]] : i64 to i32
    // CHECK: [[v44:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v45:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[v46:%.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[v47:%.*]] = llvm.inttoptr [[v46]] : i64 to !llvm.ptr
    // CHECK: [[v48:%.*]] = llvm.call @MPI_Recv([[v41]], [[v43]], [[v44]], [[v12]], [[v12]], [[v45]], [[v47]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    %2 = mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: [[v49:%.*]] = llvm.call @MPI_Finalize() : () -> i32
    %3 = mpi.finalize : !mpi.retval

    return
  }
}
