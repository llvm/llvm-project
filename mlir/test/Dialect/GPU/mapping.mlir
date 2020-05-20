// RUN: mlir-opt -test-gpu-greedy-parallel-loop-mapping -split-input-file %s | FileCheck %s

func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                    %arg3 : index) {
  %zero = constant 0 : index
  %one = constant 1 : index
  %four = constant 4 : index
  loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%four, %four)  {
    loop.parallel (%si0, %si1) = (%zero, %zero) to (%four, %four)
                                            step (%one, %one)  {
    }
  }
  return
}

// CHECK-LABEL:   func @parallel_loop(
// CHECK:           loop.parallel 
// CHECK:             loop.parallel 
// CHECK:      {mapping = [{bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 3 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 4 : i64}]}
// CHECK:      {mapping = [{bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 0 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 1 : i64}]}
// CHECK-NOT: mapping

// -----

func @parallel_loop_4d(%arg0 : index, %arg1 : index, %arg2 : index,
                       %arg3 : index) {
  %zero = constant 0 : index
  %one = constant 1 : index
  %four = constant 4 : index
  loop.parallel (%i0, %i1, %i2, %i3) = (%zero, %zero, %zero, %zero) to (%arg0, %arg1, %arg2, %arg3)
                                       step (%four, %four, %four, %four)  {
    loop.parallel (%si0, %si1, %si2, %si3) = (%zero, %zero, %zero, %zero) to (%four, %four, %four, %four)
                                             step (%one, %one, %one, %one)  {
      loop.parallel (%ti0, %ti1, %ti2, %ti3) = (%zero, %zero, %zero, %zero) to (%four, %four, %four, %four)
                                               step (%one, %one, %one, %one)  {
      }
    }
  }
  return
}

// CHECK-LABEL:   func @parallel_loop_4d(
// CHECK:           loop.parallel 
// CHECK:             loop.parallel 
// CHECK:               loop.parallel
// CHECK:      {mapping = [{bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 6 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 6 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 6 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 6 : i64}]}
// CHECK:      {mapping = [{bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 3 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 4 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 5 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 6 : i64}]}
// CHECK:      {mapping = [{bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 0 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 1 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 2 : i64},
// CHECK-SAME:             {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 6 : i64}]}
// CHECK-NOT: mapping
