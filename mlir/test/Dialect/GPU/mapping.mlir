// RUN: mlir-opt -gpu-map-parallel-loops -split-input-file %s | FileCheck %s --check-prefix=OUTER
// RUN: mlir-opt -gpu-map-parallel-loops="mapping-policy=innermost-first" -split-input-file %s | FileCheck %s --check-prefix=INNER

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

// OUTER-LABEL:   func @parallel_loop(
// OUTER:           scf.parallel
// OUTER:             scf.parallel
// OUTER:      {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// OUTER:      {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// OUTER-NOT: mapping

// INNER-LABEL:   func @parallel_loop(
// INNER:           scf.parallel
// INNER:             scf.parallel
// INNER:      {mapping = [#gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// INNER:      {mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// INNER-NOT: mapping

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

// OUTER-LABEL:   func @parallel_loop_4d(
// OUTER:           scf.parallel
// OUTER:             scf.parallel
// OUTER:               scf.parallel
// OUTER:      {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// OUTER:      {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = thread_z, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// OUTER:      {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = block_z, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// OUTER-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// OUTER-NOT: mapping

// INNER-LABEL:   func @parallel_loop_4d(
// INNER:           scf.parallel
// INNER:             scf.parallel
// INNER:               scf.parallel
// INNER:      {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// INNER:      {mapping = [#gpu.loop_dim_map<processor = thread_z, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// INNER:      {mapping = [#gpu.loop_dim_map<processor = block_z, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// INNER-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// INNER-NOT: mapping
