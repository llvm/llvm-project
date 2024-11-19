// RUN: mlir-opt -convert-to-llvm %s | FileCheck %s

module {
  // CHECK: llvm.func @MPI_Finalize() -> i32
  // CHECK: llvm.func @MPI_Recv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: llvm.func @MPI_Send(!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  // CHECK: llvm.func @MPI_Comm_rank({{.*}}, !llvm.ptr) -> i32
  // COMM: llvm.mlir.global external @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.struct<"MPI_ABI_Comm", opaque>
  // CHECK: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32

  func.func @mpi_test(%arg0: memref<100xf32>) {
    // CHECK: [[varg0:%.*]]: !llvm.ptr, [[varg1:%.*]]: !llvm.ptr, [[varg2:%.*]]: i64, [[varg3:%.*]]: i64, [[varg4:%.*]]: i64
    // CHECK: [[v0:%.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: [[v1:%.*]] = llvm.insertvalue [[varg0]], [[v0]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v2:%.*]] = llvm.insertvalue [[varg1]], [[v1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v3:%.*]] = llvm.insertvalue [[varg2]], [[v2]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v4:%.*]] = llvm.insertvalue [[varg3]], [[v3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v5:%.*]] = llvm.insertvalue [[varg4]], [[v4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v6:%.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK-NEXT: [[v7:%.*]] = llvm.call @MPI_Init([[v6]], [[v6]]) : (!llvm.ptr, !llvm.ptr) -> i32
    // CHECK-NEXT: [[v8:%.*]] = builtin.unrealized_conversion_cast [[v7]] : i32 to !mpi.retval
    %0 = mpi.init : !mpi.retval

    // CHECK: [[v9:%.*]] = llvm.mlir.constant(1140850688 : i32) : i32
    // CHECK-NEXT: [[v10:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: [[v11:%.*]] = llvm.alloca [[v10]] x i32 : (i32) -> !llvm.ptr
    // CHECK-NEXT: [[v12:%.*]] = llvm.call @MPI_Comm_rank([[v9]], [[v11]]) : (i32, !llvm.ptr) -> i32
    // CHECK-NEXT: [[v13:%.*]] = builtin.unrealized_conversion_cast [[v12]] : i32 to !mpi.retval
    // CHECK-NEXT: [[v14:%.*]] = llvm.load [[v11]] : !llvm.ptr -> i32
    %retval, %rank = mpi.comm_rank : !mpi.retval, i32

    // CHECK: [[v15:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v16:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v17:%.*]] = llvm.getelementptr [[v15]][[[v16]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK-NEXT: [[v18:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v19:%.*]] = llvm.trunc [[v18]] : i64 to i32
    // CHECK-NEXT: [[v20:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK-NEXT: [[v21:%.*]] = llvm.mlir.constant(1140850688 : i32) : i32
    // CHECK-NEXT: [[v22:%.*]] = llvm.call @MPI_Send([[v17]], [[v19]], [[v20]], [[v14]], [[v14]], [[v21]]) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
    mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK: [[v23:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v24:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v25:%.*]] = llvm.getelementptr [[v23]][[[v24]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK-NEXT: [[v26:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v27:%.*]] = llvm.trunc [[v26]] : i64 to i32
    // CHECK-NEXT: [[v28:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK-NEXT: [[v29:%.*]] = llvm.mlir.constant(1140850688 : i32) : i32
    // CHECK-NEXT: [[v30:%.*]] = llvm.call @MPI_Send([[v25]], [[v27]], [[v28]], [[v14]], [[v14]], [[v29]]) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
    %1 = mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: [[v31:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v32:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v33:%.*]] = llvm.getelementptr [[v31]][[[v32]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK-NEXT: [[v34:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v35:%.*]] = llvm.trunc [[v34]] : i64 to i32
    // CHECK-NEXT: [[v36:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK-NEXT: [[v37:%.*]] = llvm.mlir.constant(1140850688 : i32) : i32
    // CHECK-NEXT: [[v38:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT: [[v39:%.*]] = llvm.inttoptr [[v38]] : i64 to !llvm.ptr
    // CHECK-NEXT: [[v40:%.*]] = llvm.call @MPI_Recv([[v33]], [[v35]], [[v36]], [[v14]], [[v14]], [[v37]], [[v39]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
    mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK: [[v41:%.*]] = llvm.extractvalue [[v5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v42:%.*]] = llvm.extractvalue [[v5]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v43:%.*]] = llvm.getelementptr [[v41]][[[v42]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK-NEXT: [[v44:%.*]] = llvm.extractvalue [[v5]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v45:%.*]] = llvm.trunc [[v44]] : i64 to i32
    // CHECK-NEXT: [[v46:%.*]] = llvm.mlir.constant(1275069450 : i32) : i32
    // CHECK-NEXT: [[v47:%.*]] = llvm.mlir.constant(1140850688 : i32) : i32
    // CHECK-NEXT: [[v48:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT: [[v49:%.*]] = llvm.inttoptr [[v48]] : i64 to !llvm.ptr
    // CHECK-NEXT: [[v50:%.*]] = llvm.call @MPI_Recv([[v43]], [[v45]], [[v46]], [[v14]], [[v14]], [[v47]], [[v49]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
    %2 = mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: llvm.call @MPI_Finalize() : () -> i32
    %3 = mpi.finalize : !mpi.retval

    %4 = mpi.retval_check %retval = <MPI_SUCCESS> : i1

    %5 = mpi.error_class %0 : !mpi.retval
    return
  }
}
