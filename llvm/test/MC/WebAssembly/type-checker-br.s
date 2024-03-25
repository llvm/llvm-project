# RUN: llvm-mc -triple=wasm32 %s 2>&1

br_block:
  .functype br_block () -> ()
  block f32
    block i32
      f32.const 1.0
      br 1
      drop
      i32.const 1
    end_block
    drop
    f32.const 1.0
  end_block
  drop
  end_function

br_func:
  .functype br_block () -> ()
  block i32
    br 1
    i32.const 1
  end_block
  drop
  end_function
