# RUN: not llvm-mc -triple=wasm32-unknown-unknown < %s | FileCheck %s

# This tests annotations are handled correctly even if there is a type checking
# error (which are unrelated to the block annotations).
test_annotation:
  .functype   test_annotation () -> ()
  block (i32) -> ()
    drop
# CHECK-NOT: # End marker mismatch!
  end_block
  end_function
