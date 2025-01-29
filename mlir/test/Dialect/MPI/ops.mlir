// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func.func @mpi_test(%ref : memref<100xf32>) -> () {
    // Note: the !mpi.retval result is optional on all operations except mpi.error_class

    // CHECK: %0 = mpi.init : !mpi.retval
    %err = mpi.init : !mpi.retval

    // CHECK-NEXT: %retval, %rank = mpi.comm_rank : i32, !mpi.retval
    %retval, %rank = mpi.comm_rank : i32, !mpi.retval

    // CHECK-NEXT: %retval2, %size = mpi.comm_size : i32, !mpi.retval
    %retval2, %size = mpi.comm_size : i32, !mpi.retval

    // CHECK-NEXT: %comm = mpi.comm_world : mpi.comm
    %comm = mpi.comm_world : mpi.comm

    // CHECK-NEXT: %new_comm, %retval3 = mpi.comm_split(%comm, %rank, %rank) : i32, !mpi.retval
    %new_comm, %retval3 = mpi.comm_split(%comm, %rank, %rank) : mpi.comm, i32, i32

    // CHECK-NEXT: mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32
    mpi.send(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: %1 = mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval
    %err2 = mpi.send(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK-NEXT: mpi.send(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32, mpi.comm
    mpi.send(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32, mpi.comm

    // CHECK-NEXT: mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32
    mpi.recv(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: %2 = mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval
    %err3 = mpi.recv(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK-NEXT: mpi.recv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32, mpi.comm
    mpi.recv(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32, mpi.comm

    // CHECK-NEXT: %3 = mpi.isend(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request
    %req1 = mpi.isend(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request

    // CHECK-NEXT: %4, %5 = mpi.isend(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request, !mpi.retval
    %req2, %err4, = mpi.isend(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request, !mpi.retval

    // CHECK-NEXT: %3 = mpi.isend(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32, mpi.comm -> mpi.request
    %req1 = mpi.isend(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32, mpi.comm -> mpi.request

    // CHECK-NEXT: %6 = mpi.irecv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request
    %req3 = mpi.irecv(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request

    // CHECK-NEXT: %7, %8 = mpi.irecv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request, !mpi.retval
    %req4, %err5 = mpi.irecv(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request, !mpi.retval

    // CHECK-NEXT: %6 = mpi.irecv(%arg0, %rank, %rank, %comm) : memref<100xf32>, i32, i32, mpi.comm -> mpi.request
    %req3 = mpi.irecv(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32, mpi.comm -> mpi.request

    // CHECK-NEXT: mpi.wait(%req1) : mpi.request
    mpi.wait(%req1) : mpi.request

    // CHECK-NEXT: %9 = mpi.wait(%req1) : mpi.request -> !mpi.retval
    %err6 = mpi.wait(%req2) : mpi.request -> !mpi.retval

    // CHECK-NEXT: mpi.barrier : !mpi.retval
    mpi.barrier : !mpi.retval

    // CHECK-NEXT: %10 = mpi.barrier : !mpi.retval
    %err7 = mpi.barrier : !mpi.retval

    // CHECK-NEXT: mpi.barrier(%comm) : !mpi.retval
    mpi.barrier(%comm) : !mpi.retval

    // CHECK-NEXT: mpi.allreduce(%arg0, %arg0, MPI_SUM) : memref<100xf32>, memref<100xf32>
    mpi.allreduce(%ref, %ref, MPI_SUM) : memref<100xf32>, memref<100xf32>

    // CHECK-NEXT: mpi.allreduce(%arg0, %arg0, MPI_SUM) : memref<100xf32>, memref<100xf32> -> !mpi.retval
    %err8 = mpi.allreduce(%ref, %ref, MPI_SUM) : memref<100xf32>, memref<100xf32>

    // CHECK-NEXT: mpi.allreduce(%arg0, %arg0, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>, mpi.comm
    mpi.allreduce(%ref, %ref, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>, mpi.comm

    // CHECK-NEXT: %11 = mpi.finalize : !mpi.retval
    %rval = mpi.finalize : !mpi.retval

    // CHECK-NEXT: %12 = mpi.retval_check %retval = <MPI_SUCCESS> : i1
    %res = mpi.retval_check %retval = <MPI_SUCCESS> : i1

    // CHECK-NEXT: %13 = mpi.error_class %0 : !mpi.retval
    %errclass = mpi.error_class %err : !mpi.retval

    // CHECK-NEXT: return
    func.return
}
