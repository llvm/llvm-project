// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func.func @mpi_test(%ref : memref<100xf32>) -> () {
    // Note: the !mpi.retval result is optional on all operations except mpi.error_class

    // CHECK: %0 = mpi.init : !mpi.retval
    %err = mpi.init : !mpi.retval

    // CHECK-NEXT: %comm = mpi.comm_world : !mpi.comm
    %comm = mpi.comm_world : !mpi.comm

    // CHECK-NEXT: %rank = mpi.comm_rank : i32
    %rank = mpi.comm_rank : i32

    // CHECK-NEXT: %retval, %rank = mpi.comm_rank : !mpi.retval, i32
    %retval, %rank = mpi.comm_rank : !mpi.retval, i32

    // CHECK-NEXT: %retval, %rank = mpi.comm_rank : !mpi.comm -> i32
    %rank = mpi.comm_rank(%comm) : !mpi.comm -> i32

    // CHECK-NEXT: %retval, %rank = mpi.comm_rank : !mpi.comm -> !mpi.retval, i32
    %retval, %rank = mpi.comm_rank(%comm) : !mpi.comm -> !mpi.retval, i32

    // CHECK-NEXT: %size = mpi.comm_size : i32
    %size = mpi.comm_size : i32

    // CHECK-NEXT: %retval_0, %size = mpi.comm_size : !mpi.retval, i32
    %retval_0, %size = mpi.comm_size : !mpi.retval, i32

    // CHECK-NEXT: %size = mpi.comm_size : !mpi.comm -> i32
    %size = mpi.comm_size(%comm) : !mpi.comm -> i32

    // CHECK-NEXT: %retval_0, %size = mpi.comm_size : !mpi.retval, i32
    %retval_0, %size = mpi.comm_size(%comm) : !mpi.comm -> !mpi.retval, i32

    // CHECK-NEXT: %new_comm, %retval3 = mpi.comm_split(%comm, %rank, %rank) : i32, !mpi.retval
    %new_comm, %retval3 = mpi.comm_split(%comm, %rank, %rank) : mpi.comm, i32, i32

    // CHECK-NEXT: mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32
    mpi.send(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: %1 = mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval
    %err2 = mpi.send(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK-NEXT: mpi.send(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32, !mpi.comm
    mpi.send(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32, !mpi.comm

    // CHECK-NEXT: mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32
    mpi.recv(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: %2 = mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval
    %err3 = mpi.recv(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK-NEXT: mpi.recv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32, !mpi.comm
    mpi.recv(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32, !mpi.comm

    // CHECK-NEXT: %req = mpi.isend(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.request
    %req = mpi.isend(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.request

    // CHECK-NEXT: %retval_1, %req_2 = mpi.isend(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval, !mpi.request
    %err4, %req2 = mpi.isend(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval, !mpi.request

    // CHECK-NEXT: %3 = mpi.isend(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32, mpi.comm -> !mpi.request
    %req1 = mpi.isend(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32, !mpi.comm -> !mpi.request

    // CHECK-NEXT: %req_3 = mpi.irecv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.request
    %req3 = mpi.irecv(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.request

    // CHECK-NEXT: %retval_4, %req_5 = mpi.irecv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval, !mpi.request
    %err5, %req4 = mpi.irecv(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval, !mpi.request

    // CHECK-NEXT: %6 = mpi.irecv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32, mpi.comm -> mpi.request
    %req3 = mpi.irecv(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32, !mpi.comm -> !mpi.request

    // CHECK-NEXT: mpi.wait(%req) : !mpi.request
    mpi.wait(%req) : !mpi.request

    // CHECK-NEXT: %3 = mpi.wait(%req_2) : !mpi.request -> !mpi.retval
    %err6 = mpi.wait(%req2) : !mpi.request -> !mpi.retval

    // CHECK-NEXT: mpi.barrier : !mpi.retval
    mpi.barrier : !mpi.retval

    // CHECK-NEXT: %5 = mpi.barrier : !mpi.retval
    %err7 = mpi.barrier : !mpi.retval

    // CHECK-NEXT: mpi.barrier(%comm) : !mpi.retval
    mpi.barrier(%comm) : !mpi.retval

    // CHECK-NEXT: mpi.allreduce(%arg0, %arg0, <MPI_SUM>) : memref<100xf32>, memref<100xf32>
    mpi.allreduce(%ref, %ref, <MPI_SUM>) : memref<100xf32>, memref<100xf32>

    // CHECK-NEXT: mpi.allreduce(%arg0, %arg0, <MPI_SUM>) : memref<100xf32>, memref<100xf32> -> !mpi.retval
    %err8 = mpi.allreduce(%ref, %ref, <MPI_SUM>) : memref<100xf32>, memref<100xf32> -> !mpi.retval

    // CHECK-NEXT: mpi.allreduce(%arg0, %arg0, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>, !mpi.comm
    mpi.allreduce(%ref, %ref, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>, !mpi.comm

    // CHECK-NEXT: %7 = mpi.finalize : !mpi.retval
    %rval = mpi.finalize : !mpi.retval

    // CHECK-NEXT: %8 = mpi.retval_check %retval = <MPI_SUCCESS> : i1
    %res = mpi.retval_check %retval = <MPI_SUCCESS> : i1

    // CHECK-NEXT: %9 = mpi.error_class %0 : !mpi.retval
    %errclass = mpi.error_class %err : !mpi.retval

    // CHECK-NEXT: return
    func.return
}
