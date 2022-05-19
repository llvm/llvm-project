# RUN: llvm-mc -triple=wasm32 -no-type-check %s 2>&1

# Check no errors are produced when type checking is disabled.

correctly_typed:
  .functype correctly_typed () -> (i32)
  i32.const 1
	end_function

incorrectly_typed:
  .functype incorrectly_typed () -> (i32)
  nop
	end_function
