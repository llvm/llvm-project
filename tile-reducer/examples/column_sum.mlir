// Column reduction of a row-major buffer. The compiler stages through smem.

func.func @column_sum(%in: !tr.buffer<MxKxf32>, %out: !tr.buffer<Kxf32>) {
  %col_blk = tr.program_id 0 : index
  %c128    = arith.constant 128 : index
  %m       = tr.dim %in, 0 : !tr.buffer<MxKxf32>, index
  %num     = arith.divui %m, %c128 : index
  %zero    = tr.constant 0.0 : !tr.tile<128xf32>
  %result = tr.for %mt = 0 to %num step 1
      iter_args(%acc = %zero) -> !tr.tile<128xf32> {
    %t       = tr.load %in[%mt, %col_blk]
        : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
    %partial = tr.reduce_sum %t, axis = 0
        : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
    %acc2    = tr.add %acc, %partial : !tr.tile<128xf32>
    tr.yield %acc2 : !tr.tile<128xf32>
  }
  tr.store %out[%col_blk], %result : !tr.buffer<Kxf32>, !tr.tile<128xf32>
  return
}
