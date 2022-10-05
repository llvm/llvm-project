# We had a regression where an else instruction would be omitted if it followed
# return (i.e. it was in an unreachable state).
# See: https://github.com/llvm/llvm-project/issues/56935

# RUN: llvm-mc -triple=wasm32-unknown-unknown < %s | FileCheck %s

foo:
  .functype foo () -> (i32)
  i32.const 1
  if i32
    i32.const 2
    return
  else
    i32.const 3
  end_if
  end_function

# CHECK-LABEL: foo:
# CHEKC-NEXT: .functype foo () -> (i32)
# CHEKC-NEXT: i32.const 1
# CHEKC-NEXT: if i32
# CHEKC-NEXT: i32.const 2
# CHEKC-NEXT: return
# CHEKC-NEXT: else
# CHEKC-NEXT: i32.const 3
# CHEKC-NEXT: end_if
# CHEKC-NEXT: end_function
