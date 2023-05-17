# RUN: not llvm-mc -triple=wasm32-unknown-unknown %s 2>&1 | FileCheck %s

# A Wasm function should always end with a 'end_function' pseudo instruction in
# assembly. This causes the parser to properly wrap up function info when there
# is no other instructions present.

.functype test0 () -> ()
test0:

.functype test1 () -> ()
# CHECK: [[@LINE+1]]:1: error: Unmatched block construct(s) at function end: function
test1:
  end_function

.functype test2 () -> ()
test2:
# CHECK: [[@LINE+1]]:1: error: Unmatched block construct(s) at function end: function
