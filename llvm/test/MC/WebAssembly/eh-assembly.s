# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+exception-handling < %s | FileCheck %s
# Check that it converts to .o without errors, but don't check any output:
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -mattr=+exception-handling -o %t.o < %s

  .tagtype  __cpp_exception i32
  .tagtype  __c_longjmp i32
  .functype  foo () -> ()

eh_test:
  .functype  eh_test (exnref) -> ()

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

  # try_table targeting loops
  i32.const 0
  loop (i32) -> ()
    local.get 0
    loop (exnref) -> ()
      try_table (catch __cpp_exception 1) (catch_all_ref 0)
      end_try_table
      drop
    end_loop
    drop
  end_loop
  end_function

eh_legacy_test:
  .functype  eh_legacy_test () -> ()

  # try-catch with catch, catch_all, throw, and rethrow
  try
    i32.const 3
    throw     __cpp_exception
  catch       __cpp_exception
    drop
    rethrow 0
  catch       __c_longjmp
    drop
  catch_all
    rethrow 0
  end_try

  # Nested try-catch with a rethrow
  try
    call  foo
  catch_all
    try
    catch_all
      rethrow 1
    end_try
  end_try

  # try-catch with a single return value
  try i32
    i32.const 0
  catch       __cpp_exception
  end_try
  drop

  # try-catch with a mulvivalue return
  try () -> (i32, f32)
    i32.const 0
    f32.const 0.0
  catch       __cpp_exception
    f32.const 1.0
  end_try
  drop
  drop

  # Catch-less try
  try
    call  foo
  end_try
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

# CHECK:         i32.const       0
# CHECK-NEXT:    loop            (i32) -> ()
# CHECK-NEXT:    local.get       0
# CHECK-NEXT:    loop            (exnref) -> ()
# CHECK-NEXT:    try_table        (catch __cpp_exception 1) (catch_all_ref 0)
# CHECK:         end_try_table
# CHECK-NEXT:    drop
# CHECK-NEXT:    end_loop
# CHECK-NEXT:    drop
# CHECK-NEXT:    end_loop

# CHECK:       eh_legacy_test:
# CHECK:         try
# CHECK-NEXT:    i32.const       3
# CHECK-NEXT:    throw           __cpp_exception
# CHECK-NEXT:    catch           __cpp_exception
# CHECK-NEXT:    drop
# CHECK-NEXT:    rethrow         0
# CHECK-NEXT:    catch           __c_longjmp
# CHECK-NEXT:    drop
# CHECK-NEXT:    catch_all
# CHECK-NEXT:    rethrow         0
# CHECK-NEXT:    end_try

# CHECK:         try
# CHECK-NEXT:    call    foo
# CHECK-NEXT:    catch_all
# CHECK-NEXT:    try
# CHECK-NEXT:    catch_all
# CHECK-NEXT:    rethrow         1
# CHECK-NEXT:    end_try
# CHECK-NEXT:    end_try

# CHECK:         try             i32
# CHECK-NEXT:    i32.const       0
# CHECK-NEXT:    catch           __cpp_exception
# CHECK-NEXT:    end_try
# CHECK-NEXT:    drop

# CHECK:         try             () -> (i32, f32)
# CHECK-NEXT:    i32.const       0
# CHECK-NEXT:    f32.const       0x0p0
# CHECK-NEXT:    catch           __cpp_exception
# CHECK-NEXT:    f32.const       0x1p0
# CHECK-NEXT:    end_try
# CHECK-NEXT:    drop
# CHECK-NEXT:    drop

# CHECK:         try
# CHECK-NEXT:    call    foo
# CHECK-NEXT:    end_try
