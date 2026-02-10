// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Verify that nvvm.barrier roundtrips correctly when loc() annotations use
// forward-referencing #loc aliases (defined after the module body).
// This is a regression test for a bug where parseOptionalAttribute would
// speculatively consume loc(#locN) as an attribute, failing on forward
// references instead of leaving it for the trailing location parser.

// CHECK-LABEL: @barrier_loc_forward_ref
module {
  llvm.func @barrier_loc_forward_ref(%barId : i32, %numThreads : i32, %pred : i32) {
    // CHECK: nvvm.barrier
    nvvm.barrier loc(#loc1)

    // CHECK: nvvm.barrier id = %{{.*}}
    nvvm.barrier id = %barId loc(#loc2)

    // CHECK: nvvm.barrier id = %{{.*}} number_of_threads = %{{.*}}
    nvvm.barrier id = %barId number_of_threads = %numThreads loc(#loc1)

    // CHECK: %{{.*}} = nvvm.barrier #nvvm.reduction<and> %{{.*}} -> i32
    %0 = nvvm.barrier #nvvm.reduction<and> %pred -> i32 loc(#loc2)

    // CHECK: %{{.*}} = nvvm.barrier #nvvm.reduction<popc> %{{.*}} -> i32
    %1 = nvvm.barrier #nvvm.reduction<popc> %pred -> i32 loc(#loc1)

    llvm.return
  } loc(#loc0)
} loc(#loc0)
#loc0 = loc(unknown)
#loc1 = loc("barrier_test"(#loc0))
#loc2 = loc("barrier_with_id"(#loc0))
