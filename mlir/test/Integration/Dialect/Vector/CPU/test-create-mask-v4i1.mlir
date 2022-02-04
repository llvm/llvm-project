// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// NOTE: This is similar to test-create-mask.mlir, but with a different length,
//       because the v4i1 vector specifically exposed bugs in the LLVM backend.

func @entry() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index

  //
  // 1-D.
  //

  %1 = vector.create_mask %c2 : vector<4xi1>
  vector.print %1 : vector<4xi1>
  // CHECK: ( 1, 1, 0, 0 )

  scf.for %i = %c0 to %c5 step %c1 {
    %2 = vector.create_mask %i : vector<4xi1>
    vector.print %2 : vector<4xi1>
  }
  // CHECK: ( 0, 0, 0, 0 )
  // CHECK: ( 1, 0, 0, 0 )
  // CHECK: ( 1, 1, 0, 0 )
  // CHECK: ( 1, 1, 1, 0 )
  // CHECK: ( 1, 1, 1, 1 )

  //
  // 2-D.
  //

  %3 = vector.create_mask %c2, %c3 : vector<4x4xi1>
  vector.print %3 : vector<4x4xi1>
  // CHECK: ( ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )

  %4 = vector.create_mask %c3, %c2 : vector<4x4xi1>
  vector.print %4 : vector<4x4xi1>
  // CHECK: ( ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 0, 0, 0, 0 ) )

  scf.for %i = %c0 to %c5 step %c1 {
    %5 = vector.create_mask %c2, %i : vector<4x4xi1>
    vector.print %5 : vector<4x4xi1>
  }
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 0, 0, 0 ), ( 1, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )

  scf.for %i = %c0 to %c5 step %c1 {
    %6 = vector.create_mask %i, %c2 : vector<4x4xi1>
    vector.print %6 : vector<4x4xi1>
  }
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ) )

  scf.for %i = %c0 to %c5 step %c1 {
    scf.for %j = %c0 to %c5 step %c1 {
      %7 = vector.create_mask %i, %j : vector<4x4xi1>
      vector.print %7 : vector<4x4xi1>
    }
  }
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 1, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 1, 1 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 0, 0, 0 ), ( 1, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 0, 0, 0 ), ( 1, 0, 0, 0 ), ( 1, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
  // CHECK: ( ( 1, 0, 0, 0 ), ( 1, 0, 0, 0 ), ( 1, 0, 0, 0 ), ( 1, 0, 0, 0 ) )
  // CHECK: ( ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ) )
  // CHECK: ( ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ) )
  // CHECK: ( ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ) )

  return
}
