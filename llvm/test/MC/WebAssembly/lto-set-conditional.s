# RUN: llvm-mc -triple=wasm32-unknown-unknown

# Tests if `.lto_set_conditional` directives are parsed without crashing.
.lto_set_conditional a, a.new
.type  a.new,@function
a.new:
  .functype  a.new () -> ()
  end_function
