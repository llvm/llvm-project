# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+exception-handling --no-type-check < %s | FileCheck %s

  .tagtype  __cpp_exception i32
  .tagtype  __c_longjmp i32
  .functype  foo () -> ()

eh_test:
  .functype  eh_test () -> ()

  # try_table with all four kinds of catch clauses
  block exnref
    block
      block () -> (i32, exnref)
        block i32
          try_table (catch __cpp_exception 0) (catch_ref __c_longjmp 1) (catch_all 2) (catch_all_ref 3)
            i32.const 0
            throw     __cpp_exception
          end_try_table
          return
        end_block
        drop
        return
      end_block
      throw_ref
      drop
    end_block
    return
  end_block
  drop

  # You can use the same kind of catch clause more than once
  block
    block exnref
      block
        try_table (catch_all 0) (catch_all_ref 1) (catch_all 2)
          call  foo
        end_try_table
      end_block
      return
    end_block
    drop
  end_block

  # Two catch clauses targeting the same block
  block
    try_table (catch_all 0) (catch_all 0)
    end_try_table
  end_block

  # try_table with a return type
  block
    try_table f32 (catch_all 0)
      f32.const 0.0
    end_try_table
    drop
  end_block

  # try_table with a multivalue type return
  block
    try_table () -> (i32, f32) (catch_all 0)
      i32.const 0
      f32.const 0.0
    end_try_table
    drop
    drop
  end_block

  # catch-less try_tables
  try_table
    call  foo
  end_try_table

  try_table i32
    i32.const 0
  end_try_table
  drop

  try_table () -> (i32, f32)
    i32.const 0
    f32.const 0.0
  end_try_table
  drop
  drop

  end_function

# CHECK-LABEL: eh_test:
# CHECK:         block           exnref
# CHECK-NEXT:    block
# CHECK-NEXT:    block           () -> (i32, exnref)
# CHECK-NEXT:    block           i32
# CHECK-NEXT:    try_table        (catch __cpp_exception 0) (catch_ref __c_longjmp 1) (catch_all 2) (catch_all_ref 3)
# CHECK:         i32.const       0
# CHECK-NEXT:    throw           __cpp_exception
# CHECK-NEXT:    end_try_table
# CHECK-NEXT:    return
# CHECK-NEXT:    end_block
# CHECK-NEXT:    drop
# CHECK-NEXT:    return
# CHECK-NEXT:    end_block
# CHECK-NEXT:    throw_ref
# CHECK-NEXT:    drop
# CHECK-NEXT:    end_block
# CHECK-NEXT:    return
# CHECK-NEXT:    end_block
# CHECK-NEXT:    drop

# CHECK:         block
# CHECK-NEXT:    block           exnref
# CHECK-NEXT:    block
# CHECK-NEXT:    try_table        (catch_all 0) (catch_all_ref 1) (catch_all 2)
# CHECK:         call    foo
# CHECK-NEXT:    end_try_table
# CHECK-NEXT:    end_block
# CHECK-NEXT:    return
# CHECK-NEXT:    end_block
# CHECK-NEXT:    drop
# CHECK-NEXT:    end_block

# CHECK:         block
# CHECK-NEXT:    try_table        (catch_all 0) (catch_all 0)
# CHECK:         end_try_table
# CHECK-NEXT:    end_block

# CHECK:         block
# CHECK-NEXT:    try_table       f32 (catch_all 0)
# CHECK:         f32.const       0x0p0
# CHECK-NEXT:    end_try_table
# CHECK-NEXT:    drop
# CHECK-NEXT:    end_block

# CHECK:         block
# CHECK-NEXT:    try_table       () -> (i32, f32) (catch_all 0)
# CHECK:         i32.const       0
# CHECK-NEXT:    f32.const       0x0p0
# CHECK-NEXT:    end_try_table
# CHECK-NEXT:    drop
# CHECK-NEXT:    drop
# CHECK-NEXT:    end_block

# CHECK:         try_table
# CHECK-NEXT:    call    foo
# CHECK-NEXT:    end_try_table

# CHECK:         try_table       i32
# CHECK-NEXT:    i32.const       0
# CHECK-NEXT:    end_try_table
# CHECK-NEXT:    drop

# CHECK:         try_table       () -> (i32, f32)
# CHECK-NEXT:    i32.const       0
# CHECK-NEXT:    f32.const       0x0p0
# CHECK-NEXT:    end_try_table
# CHECK-NEXT:    drop
# CHECK-NEXT:    drop
