# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+exception-handling < %s | FileCheck %s
# Check that it converts to .o without errors, but don't check any output:
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+exception-handling -filetype=obj -o %t.o < %s

  .tagtype  __cpp_exception i32

block_branch_test:
  .functype  block_branch_test () -> ()

  # Block input paramter / return tests

  i32.const 0
  block (i32) -> (i32)
  end_block
  drop

  i32.const 0
  i64.const 0
  block (i32, i64) -> (i32, f32)
    drop
    f32.const 0.0
  end_block
  drop
  drop

  i32.const 0
  loop (i32) -> (f32)
    drop
    f32.const 0.0
  end_loop
  drop

  i32.const 0
  i32.const 0
  if (i32) -> (i32)
  else
    i32.popcnt
  end_if
  drop

  try i32
    i32.const 0
  catch       __cpp_exception
    i32.clz
  catch_all
    i32.const 5
  end_try
  drop

  i32.const 0
  block (i32) -> (i32)
    block (i32) -> (f32)
      drop
      f32.const 0.0
    end_block
    drop
    i32.const 0
  end_block
  drop

  # Branch tests

  block f32
    f32.const 0.0
    i32.const 0
    br_if 0
    f32.const 1.0
    br 0
    # After 'br', we can pop any values from the polymorphic stack
    i32.add
    i32.sub
    i32.mul
    drop
  end_block
  drop

  block () -> (f32, f64)
    f32.const 0.0
    f64.const 0.0
    i32.const 0
    br_if 0
    block (f32, f64) -> (f32, f64)
      i32.const 1
      br_if 0
    end_block
  end_block
  drop
  drop

  # Within a loop, branches target the start of the loop
  i32.const 0
  loop (i32) -> ()
    i32.const 1
    br 0
  end_loop

  end_function

# CHECK-LABEL: block_branch_test

# CHECK:         i32.const  0
# CHECK-NEXT:    block     (i32) -> (i32)
# CHECK-NEXT:    end_block                               # label0:
# CHECK-NEXT:    drop

# CHECK:         i32.const  0
# CHECK-NEXT:    i64.const  0
# CHECK-NEXT:    block     (i32, i64) -> (i32, f32)
# CHECK-NEXT:    drop
# CHECK-NEXT:    f32.const  0x0p0
# CHECK-NEXT:    end_block                               # label1:
# CHECK-NEXT:    drop
# CHECK-NEXT:    drop

# CHECK:         i32.const  0
# CHECK-NEXT:    loop      (i32) -> (f32)                  # label2:
# CHECK-NEXT:    drop
# CHECK-NEXT:    f32.const  0x0p0
# CHECK-NEXT:    end_loop
# CHECK-NEXT:    drop

# CHECK:         i32.const  0
# CHECK-NEXT:    i32.const  0
# CHECK-NEXT:    if      (i32) -> (i32)
# CHECK-NEXT:    else
# CHECK-NEXT:    i32.popcnt
# CHECK-NEXT:    end_if
# CHECK-NEXT:    drop

# CHECK:         try       i32
# CHECK-NEXT:    i32.const  0
# CHECK-NEXT:    catch     __cpp_exception                 # catch3:
# CHECK-NEXT:    i32.clz
# CHECK-NEXT:    catch_all
# CHECK-NEXT:    i32.const  5
# CHECK-NEXT:    end_try                                 # label3:
# CHECK-NEXT:    drop

# CHECK:         i32.const  0
# CHECK-NEXT:    block     (i32) -> (i32)
# CHECK-NEXT:    block     (i32) -> (f32)
# CHECK-NEXT:    drop
# CHECK-NEXT:    f32.const  0x0p0
# CHECK-NEXT:    end_block                               # label5:
# CHECK-NEXT:    drop
# CHECK-NEXT:    i32.const  0
# CHECK-NEXT:    end_block                               # label4:
# CHECK-NEXT:    drop

# CHECK:         block     f32
# CHECK-NEXT:    f32.const  0x0p0
# CHECK-NEXT:    i32.const  0
# CHECK-NEXT:    br_if     0                               # 0: down to label6
# CHECK-NEXT:    f32.const  0x1p0
# CHECK-NEXT:    br        0                               # 0: down to label6
# CHECK-NEXT:    i32.add
# CHECK-NEXT:    i32.sub
# CHECK-NEXT:    i32.mul
# CHECK-NEXT:    drop
# CHECK-NEXT:    end_block                               # label6:
# CHECK-NEXT:    drop

# CHECK:         block     () -> (f32, f64)
# CHECK-NEXT:    f32.const  0x0p0
# CHECK-NEXT:    f64.const  0x0p0
# CHECK-NEXT:    i32.const  0
# CHECK-NEXT:    br_if     0                               # 0: down to label7
# CHECK-NEXT:    block     (f32, f64) -> (f32, f64)
# CHECK-NEXT:    i32.const  1
# CHECK-NEXT:    br_if     0                               # 0: down to label8
# CHECK-NEXT:    end_block                               # label8:
# CHECK-NEXT:    end_block                               # label7:
# CHECK-NEXT:    drop
# CHECK-NEXT:    drop

# CHECK:         i32.const  0
# CHECK-NEXT:    loop      (i32) -> ()                     # label9:
# CHECK-NEXT:    i32.const  1
# CHECK-NEXT:    br        0                               # 0: up to label9
# CHECK-NEXT:    end_loop

# CHECK:         end_function
