// Full MxK -> scalar. The compiler emits @full_sum_stage1 / @full_sum_stage2.

func.func @full_sum(%in: !tr.buffer<MxKxf32>, %out: !tr.buffer<1xf32>) {
  %blk  = tr.program_id 0 : index
  %c128 = arith.constant 128 : index
  %c0   = arith.constant 0 : index
  %k    = tr.dim %in, 1 : !tr.buffer<MxKxf32>, index
  %num  = arith.divui %k, %c128 : index
  %zero = tr.constant 0.0 : !tr.tile<f32>
  %result = tr.for %kt = 0 to %num step 1
      iter_args(%acc = %zero) -> !tr.tile<f32> {
    %t    = tr.load %in[%blk, %kt]
        : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
    %row  = tr.reduce_sum %t, axis = 1
        : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
    %s    = tr.reduce_sum %row, axis = 0
        : !tr.tile<128xf32> -> !tr.tile<f32>
    %acc2 = tr.add %acc, %s : !tr.tile<f32>
    tr.yield %acc2 : !tr.tile<f32>
  }
  tr.store %out[%c0], %result : !tr.buffer<1xf32>, !tr.tile<f32>
  return
}
