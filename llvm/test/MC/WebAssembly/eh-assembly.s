# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+exception-handling < %s | FileCheck %s
# Check that it converts to .o without errors, but don't check any output:
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -mattr=+exception-handling -o %t.o < %s

  .tagtype  __cpp_exception i32
  .tagtype  __c_longjmp i32
  .functype  eh_legacy_test () -> ()
  .functype  foo () -> ()

eh_legacy_test:
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
  end_function

# CHECK-LABEL: eh_legacy_test:
# CHECK-NEXT:    try
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
# CHECK-NEXT:    end_function
