# RUN: llvm-mc -triple=wasm32 -mattr=+tail-call %s 2>&1

# XFAIL: *

# FIXME: These shouldn't produce an error, as return will implicitly drop any
# superfluous values.

return_superfluous_return_values:
  .functype return_superfluous_return_values () -> ()
  i32.const 1
  f32.const 2.0
  return
  end_function

return_call_indirect_superfluous_return_values:
  .functype return_call_indirect_superfluous_return_values () -> ()
  f32.const 1.0
  i32.const 2
  return_call_indirect () -> ()
  end_function

.functype fn_void_to_void () -> ()

return_call_superfluous_return_values:
  .functype return_call_superfluous_return_values () -> ()
  f32.const 1.0
  i32.const 2
  return_call fn_void_to_void
  end_function
