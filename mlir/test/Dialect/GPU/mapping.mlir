// RUN: mlir-opt -test-gpu-greedy-parallel-loop-mapping -split-input-file %s | FileCheck %s

func.func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                    %arg3 : index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%four, %four)  {
    scf.parallel (%si0, %si1) = (%zero, %zero) to (%four, %four)
                                            step (%one, %one)  {
    }
  }
  return
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func @parallel_loop(
// CHECK:           scf.parallel
// CHECK:             scf.parallel
// CHECK:      {mapping = [{bound = #[[$MAP]], map = #[[$MAP]], processor = 3 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 4 : i64}]}
// CHECK:      {mapping = [{bound = #[[$MAP]], map = #[[$MAP]], processor = 0 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 1 : i64}]}
// CHECK-NOT: mapping

// -----

func.func @parallel_loop_4d(%arg0 : index, %arg1 : index, %arg2 : index,
                       %arg3 : index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.parallel (%i0, %i1, %i2, %i3) = (%zero, %zero, %zero, %zero) to (%arg0, %arg1, %arg2, %arg3)
                                       step (%four, %four, %four, %four)  {
    scf.parallel (%si0, %si1, %si2, %si3) = (%zero, %zero, %zero, %zero) to (%four, %four, %four, %four)
                                             step (%one, %one, %one, %one)  {
      scf.parallel (%ti0, %ti1, %ti2, %ti3) = (%zero, %zero, %zero, %zero) to (%four, %four, %four, %four)
                                               step (%one, %one, %one, %one)  {
      }
    }
  }
  return
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func @parallel_loop_4d(
// CHECK:           scf.parallel
// CHECK:             scf.parallel
// CHECK:               scf.parallel
// CHECK:      {mapping = [{bound = #[[$MAP]], map = #[[$MAP]], processor = 6 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 6 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 6 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 6 : i64}]}
// CHECK:      {mapping = [{bound = #[[$MAP]], map = #[[$MAP]], processor = 3 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 4 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 5 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 6 : i64}]}
// CHECK:      {mapping = [{bound = #[[$MAP]], map = #[[$MAP]], processor = 0 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 1 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 2 : i64},
// CHECK-SAME:             {bound = #[[$MAP]], map = #[[$MAP]], processor = 6 : i64}]}
// CHECK-NOT: mapping
