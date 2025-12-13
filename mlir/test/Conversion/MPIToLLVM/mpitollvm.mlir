// RUN: mlir-opt -split-input-file -convert-to-llvm %s | FileCheck %s

// COM: Test MPICH ABI
// CHECK: module attributes {dlti.map = #dlti.map<"MPI:Implementation" = "MPICH">} {
// CHECK: llvm.func @MPI_Finalize() -> i32
// CHECK: llvm.func @MPI_Comm_split(i32, i32, i32, !llvm.ptr) -> i32
// CHECK: llvm.func @MPI_Recv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK: llvm.func @MPI_Send(!llvm.ptr, i32, i32, i32, i32, i32) -> i32
// CHECK: llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
// CHECK: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
module attributes {dlti.map = #dlti.map<"MPI:Implementation" = "MPICH">} {

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

    // CHECK: [[comm:%.*]] = llvm.mlir.constant(1140850688 : i64) : i64
    %comm = mpi.comm_world : !mpi.comm

    // CHECK: [[v8:%.*]] = llvm.trunc [[comm]] : i64 to i32
    // CHECK: [[v9:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[v10:%.*]] = llvm.alloca [[v9]] x i32 : (i32) -> !llvm.ptr
    // CHECK: [[v11:%.*]] = llvm.call @MPI_Comm_rank([[v8]], [[v10]]) : (i32, !llvm.ptr) -> i32
    %retval, %rank = mpi.comm_rank(%comm) : !mpi.retval, i32

    // CHECK: [[v12:%.*]] = llvm.load [[v10]] : !llvm.ptr -> i32
    // CHECK: [[v13:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v14:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v15:%.*]] = llvm.getelementptr [[v13]][[[v14]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v16:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v17:%.*]] = llvm.trunc [[v16]] : i64 to i32
    // CHECK: [[v18:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[comm_1:%.*]] = llvm.trunc [[comm]] : i64 to i32
    // CHECK: [[v20:%.*]] = llvm.call @MPI_Send([[v15]], [[v17]], [[v18]], [[v12]], [[v12]], [[comm_1]]) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
    mpi.send(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32

    // CHECK: [[v21:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v22:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v23:%.*]] = llvm.getelementptr [[v21]][[[v22]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v24:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v25:%.*]] = llvm.trunc [[v24]] : i64 to i32
    // CHECK: [[v26:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[comm_2:%.*]] = llvm.trunc [[comm]] : i64 to i32
    // CHECK: [[v28:%.*]] = llvm.call @MPI_Send([[v23]], [[v25]], [[v26]], [[v12]], [[v12]], [[comm_2]]) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
    %1 = mpi.send(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: [[v29:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v30:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v31:%.*]] = llvm.getelementptr [[v29]][[[v30]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v32:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v33:%.*]] = llvm.trunc [[v32]] : i64 to i32
    // CHECK: [[v34:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[comm_3:%.*]] = llvm.trunc [[comm]] : i64 to i32
    // CHECK: [[v36:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[v37:%.*]] = llvm.inttoptr [[v36]] : i64 to !llvm.ptr
    // CHECK: [[v38:%.*]] = llvm.call @MPI_Recv([[v31]], [[v33]], [[v34]], [[v12]], [[v12]], [[comm_3]], [[v37]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
    mpi.recv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32

    // CHECK: [[v39:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v40:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v41:%.*]] = llvm.getelementptr [[v39]][[[v40]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v42:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v43:%.*]] = llvm.trunc [[v42]] : i64 to i32
    // CHECK: [[v44:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[comm_4:%.*]] = llvm.trunc [[comm]] : i64 to i32
    // CHECK: [[v46:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[v47:%.*]] = llvm.inttoptr [[v46]] : i64 to !llvm.ptr
    // CHECK: [[v48:%.*]] = llvm.call @MPI_Recv([[v41]], [[v43]], [[v44]], [[v12]], [[v12]], [[comm_4]], [[v47]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
    %2 = mpi.recv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval
    
    // CHECK: [[v51:%.*]] = llvm.mlir.constant(10 : i32) : i32
    %color = arith.constant 10 : i32
    // CHECK: [[v52:%.*]] = llvm.mlir.constant(22 : i32) : i32
    %key = arith.constant 22 : i32
    // CHECK: [[v53:%.*]] = llvm.trunc [[comm]] : i64 to i32
    // CHECK: [[v54:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[v55:%.*]] = llvm.alloca [[v54]] x i32 : (i32) -> !llvm.ptr
    // CHECK: [[v56:%.*]] = llvm.call @MPI_Comm_split([[v53]], [[v51]], [[v52]], [[v55]]) : (i32, i32, i32, !llvm.ptr) -> i32
    // CHECK: [[v57:%.*]] = llvm.load [[v55]] : !llvm.ptr -> i32
    %split = mpi.comm_split(%comm, %color, %key) : !mpi.comm

    // CHECK: [[v59:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v60:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v61:%.*]] = llvm.getelementptr [[v59]][[[v60]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v62:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v63:%.*]] = llvm.trunc [[v62]] : i64 to i32
    // CHECK: [[v64:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v65:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v66:%.*]] = llvm.getelementptr [[v64]][[[v65]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v67:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v68:%.*]] = llvm.trunc [[v67]] : i64 to i32
    // CHECK: [[ip:%.*]] = llvm.mlir.constant(-1 : i64) : i64
    // CHECK: [[ipp:%.*]] = llvm.inttoptr [[ip]] : i64 to !llvm.ptr
    // CHECK: [[v69:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK: [[v70:%.*]] = llvm.mlir.constant(1476395011 : i32) : i32
    // CHECK: [[v71:%.*]] = llvm.trunc [[comm]] : i64 to i32
    // CHECK: [[v72:%.*]] = llvm.call @MPI_Allreduce([[ipp]], [[v66]], [[v63]], [[v69]], [[v70]], [[v71]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
    mpi.allreduce(%arg0, %arg0, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>

    // CHECK: llvm.call @MPI_Finalize() : () -> i32
    %3 = mpi.finalize : !mpi.retval

    return
  }
}

// -----

// COM: Test OpenMPI ABI
// CHECK: module attributes {dlti.map = #dlti.map<"MPI:Implementation" = "OpenMPI">} {
// CHECK: llvm.func @MPI_Finalize() -> i32
// CHECK: llvm.func @MPI_Comm_split(!llvm.ptr, i32, i32, !llvm.ptr) -> i32
// CHECK: llvm.func @MPI_Recv(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.func @MPI_Send(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
// CHECK: llvm.mlir.global external @ompi_mpi_float() {addr_space = 0 : i32} : !llvm.struct<"ompi_predefined_datatype_t", opaque>
// CHECK: llvm.func @MPI_Comm_rank(!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.mlir.global external @ompi_mpi_comm_world() {addr_space = 0 : i32} : !llvm.struct<"ompi_communicator_t", opaque>
// CHECK: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
module attributes { dlti.map = #dlti.map<"MPI:Implementation" = "OpenMPI"> } {

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

    %comm = mpi.comm_world : !mpi.comm
    // CHECK: [[v8:%.*]] = llvm.mlir.addressof @ompi_mpi_comm_world : !llvm.ptr
    // CHECK: [[comm:%.*]] = llvm.ptrtoint [[v8]] : !llvm.ptr to i64
    // CHECK: [[comm_1:%.*]] = llvm.inttoptr [[comm]] : i64 to !llvm.ptr
    // CHECK: [[v9:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[v10:%.*]] = llvm.alloca [[v9]] x i32 : (i32) -> !llvm.ptr
    // CHECK: [[v11:%.*]] = llvm.call @MPI_Comm_rank([[comm_1]], [[v10]]) : (!llvm.ptr, !llvm.ptr) -> i32
    %retval, %rank = mpi.comm_rank(%comm) : !mpi.retval, i32

    // CHECK: [[v12:%.*]] = llvm.load [[v10]] : !llvm.ptr -> i32
    // CHECK: [[v13:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v14:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v15:%.*]] = llvm.getelementptr [[v13]][[[v14]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v16:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v17:%.*]] = llvm.trunc [[v16]] : i64 to i32
    // CHECK: [[v18:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v19:%.*]] = llvm.inttoptr [[comm]] : i64 to !llvm.ptr
    // CHECK: [[v20:%.*]] = llvm.call @MPI_Send([[v15]], [[v17]], [[v18]], [[v12]], [[v12]], [[v19]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
    mpi.send(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32

    // CHECK: [[v21:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v22:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v23:%.*]] = llvm.getelementptr [[v21]][[[v22]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v24:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v25:%.*]] = llvm.trunc [[v24]] : i64 to i32
    // CHECK: [[v26:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v27:%.*]] = llvm.inttoptr [[comm]] : i64 to !llvm.ptr
    // CHECK: [[v28:%.*]] = llvm.call @MPI_Send([[v23]], [[v25]], [[v26]], [[v12]], [[v12]], [[v27]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
    %1 = mpi.send(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: [[v29:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v30:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v31:%.*]] = llvm.getelementptr [[v29]][[[v30]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v32:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v33:%.*]] = llvm.trunc [[v32]] : i64 to i32
    // CHECK: [[v34:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v35:%.*]] = llvm.inttoptr [[comm]] : i64 to !llvm.ptr
    // CHECK: [[v36:%.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[v37:%.*]] = llvm.inttoptr [[v36]] : i64 to !llvm.ptr
    // CHECK: [[v38:%.*]] = llvm.call @MPI_Recv([[v31]], [[v33]], [[v34]], [[v12]], [[v12]], [[v35]], [[v37]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    mpi.recv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32

    // CHECK: [[v39:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v40:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v41:%.*]] = llvm.getelementptr [[v39]][[[v40]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v42:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v43:%.*]] = llvm.trunc [[v42]] : i64 to i32
    // CHECK: [[v44:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v45:%.*]] = llvm.inttoptr [[comm]] : i64 to !llvm.ptr
    // CHECK: [[v46:%.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[v47:%.*]] = llvm.inttoptr [[v46]] : i64 to !llvm.ptr
    // CHECK: [[v48:%.*]] = llvm.call @MPI_Recv([[v41]], [[v43]], [[v44]], [[v12]], [[v12]], [[v45]], [[v47]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    %2 = mpi.recv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: [[v49:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v50:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v51:%.*]] = llvm.getelementptr [[v49]][[[v50]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v52:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v53:%.*]] = llvm.trunc [[v52]] : i64 to i32
    // CHECK: [[v54:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v55:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v56:%.*]] = llvm.getelementptr [[v54]][[[v55]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: [[v57:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[v58:%.*]] = llvm.trunc [[v57]] : i64 to i32
    // CHECK: [[ip:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[ipp:%.*]] = llvm.inttoptr [[ip]] : i64 to !llvm.ptr
    // CHECK: [[v59:%.*]] = llvm.mlir.addressof @ompi_mpi_float : !llvm.ptr
    // CHECK: [[v60:%.*]] = llvm.mlir.addressof @ompi_mpi_sum : !llvm.ptr
    // CHECK: [[v61:%.*]] = llvm.inttoptr [[comm]] : i64 to !llvm.ptr
    // CHECK: [[v62:%.*]] = llvm.call @MPI_Allreduce([[ipp]], [[v56]], [[v53]], [[v59]], [[v60]], [[v61]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    mpi.allreduce(%arg0, %arg0, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>

    // CHECK: [[v71:%.*]] = llvm.mlir.constant(10 : i32) : i32
    %color = arith.constant 10 : i32
    // CHECK: [[v72:%.*]] = llvm.mlir.constant(22 : i32) : i32
    %key = arith.constant 22 : i32
    // CHECK: [[v73:%.*]] = llvm.inttoptr [[comm]] : i64 to !llvm.ptr
    // CHECK: [[v74:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[v75:%.*]] = llvm.alloca [[v74]] x !llvm.ptr : (i32) -> !llvm.ptr
    // CHECK: [[v76:%.*]] = llvm.call @MPI_Comm_split([[v73]], [[v71]], [[v72]], [[v75]]) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i32
    // CHECK: [[v77:%.*]] = llvm.load [[v75]] : !llvm.ptr -> i32
    %split = mpi.comm_split(%comm, %color, %key) : !mpi.comm

    // CHECK: llvm.call @MPI_Finalize() : () -> i32
    %3 = mpi.finalize : !mpi.retval

    return
  }
}
