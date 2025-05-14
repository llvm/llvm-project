// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @mpi_test(
// CHECK-SAME: [[varg0:%.*]]: memref<100xf32>) {
func.func @mpi_test(%ref : memref<100xf32>) -> () {
    // Note: the !mpi.retval result is optional on all operations except mpi.error_class

    // CHECK-NEXT: [[v0:%.*]] = mpi.init : !mpi.retval
    %err = mpi.init : !mpi.retval

    // CHECK-NEXT: [[v1:%.*]] = mpi.comm_world : !mpi.comm
    %comm = mpi.comm_world : !mpi.comm

    // CHECK-NEXT: [[vrank:%.*]] = mpi.comm_rank([[v1]]) : i32
    %rank = mpi.comm_rank(%comm) : i32

    // CHECK-NEXT: [[vretval:%.*]], [[vrank_0:%.*]] = mpi.comm_rank([[v1]]) : !mpi.retval, i32
    %retval, %rank_1 = mpi.comm_rank(%comm) : !mpi.retval, i32

    // CHECK-NEXT: [[vsize:%.*]] = mpi.comm_size([[v1]]) : i32
    %size = mpi.comm_size(%comm) : i32

    // CHECK-NEXT: [[vretval_1:%.*]], [[vsize_2:%.*]] = mpi.comm_size([[v1]]) : !mpi.retval, i32
    %retval_0, %size_1 = mpi.comm_size(%comm) : !mpi.retval, i32

    // CHECK-NEXT: [[vnewcomm:%.*]] = mpi.comm_split([[v1]], [[vrank]], [[vrank]]) : !mpi.comm
    %new_comm = mpi.comm_split(%comm, %rank, %rank) : !mpi.comm

    // CHECK-NEXT: [[vretval_3:%.*]], [[vnewcomm_4:%.*]] = mpi.comm_split([[v1]], [[vrank]], [[vrank]]) : !mpi.retval, !mpi.comm
    %retval_1, %new_comm_1 = mpi.comm_split(%comm, %rank, %rank) : !mpi.retval, !mpi.comm

    // CHECK-NEXT: mpi.send([[varg0]], [[vrank]], [[vrank]], [[v1]]) : memref<100xf32>, i32, i32
    mpi.send(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32

    // CHECK-NEXT: [[v2:%.*]] = mpi.send([[varg0]], [[vrank]], [[vrank]], [[v1]]) : memref<100xf32>, i32, i32 -> !mpi.retval
    %retval_2 = mpi.send(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK-NEXT: mpi.recv([[varg0]], [[vrank]], [[vrank]], [[v1]]) : memref<100xf32>, i32, i32
    mpi.recv(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32

    // CHECK-NEXT: [[v3:%.*]] = mpi.recv([[varg0]], [[vrank]], [[vrank]], [[v1]]) : memref<100xf32>, i32, i32 -> !mpi.retval
    %retval_3 = mpi.recv(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK-NEXT: [[vretval_5:%.*]], [[vreq:%.*]] = mpi.isend([[varg0]], [[vrank]], [[vrank]], [[v1]]) : memref<100xf32>, i32, i32 -> !mpi.retval, !mpi.request
    %err4, %req2 = mpi.isend(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval, !mpi.request

    // CHECK-NEXT: [[vreq_6:%.*]] = mpi.isend([[varg0]], [[vrank]], [[vrank]], [[v1]]) : memref<100xf32>, i32, i32 -> !mpi.request
    %req1 = mpi.isend(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.request

    // CHECK-NEXT: [[vreq_7:%.*]] = mpi.irecv([[varg0]], [[vrank]], [[vrank]], [[v1]]) : memref<100xf32>, i32, i32 -> !mpi.request
    %req3 = mpi.irecv(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.request

    // CHECK-NEXT: [[vretval_8:%.*]], [[vreq_9:%.*]] = mpi.irecv([[varg0]], [[vrank]], [[vrank]], [[v1]]) : memref<100xf32>, i32, i32 -> !mpi.retval, !mpi.request
    %err5, %req4 = mpi.irecv(%ref, %rank, %rank, %comm) : memref<100xf32>, i32, i32 -> !mpi.retval, !mpi.request

    // CHECK-NEXT: mpi.wait([[vreq_9]]) : !mpi.request
    mpi.wait(%req4) : !mpi.request

    // CHECK-NEXT: [[v4:%.*]] = mpi.wait([[vreq]]) : !mpi.request -> !mpi.retval
    %err6 = mpi.wait(%req2) : !mpi.request -> !mpi.retval

    // CHECK-NEXT: mpi.barrier([[v1]])
    mpi.barrier(%comm)

    // CHECK-NEXT: [[v5:%.*]] = mpi.barrier([[v1]]) -> !mpi.retval
    %err7 = mpi.barrier(%comm) -> !mpi.retval

    // CHECK-NEXT: [[v6:%.*]] = mpi.allreduce([[varg0]], [[varg0]], MPI_SUM, [[v1]]) : memref<100xf32>, memref<100xf32> -> !mpi.retval
    %err8 = mpi.allreduce(%ref, %ref, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32> -> !mpi.retval

    // CHECK-NEXT: mpi.allreduce([[varg0]], [[varg0]], MPI_SUM, [[v1]]) : memref<100xf32>, memref<100xf32>
    mpi.allreduce(%ref, %ref, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>

    // CHECK-NEXT: [[v7:%.*]] = mpi.finalize : !mpi.retval
    %rval = mpi.finalize : !mpi.retval

    // CHECK-NEXT: [[v8:%.*]] = mpi.retval_check [[vretval:%.*]] = <MPI_SUCCESS> : i1
    %res = mpi.retval_check %retval = <MPI_SUCCESS> : i1

    // CHECK-NEXT: [[v9:%.*]] = mpi.error_class [[v0]] : !mpi.retval
    %errclass = mpi.error_class %err : !mpi.retval

    // CHECK-NEXT: return
    func.return
}
