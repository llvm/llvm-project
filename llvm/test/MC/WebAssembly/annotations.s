# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+exception-handling < %s | FileCheck %s

# Tests if block/loop/try/catch/end/branch/rethrow instructions are correctly
# printed with their annotations.

  .text
  .section .text.test_annotation,"",@
  .type    test_annotation,@function
test_annotation:
  .functype   test_annotation () -> ()
  .eventtype  __cpp_exception i32
  try
  br        0
  catch
  block
  br_if     0
  loop
  br_if     1
  end_loop
  end_block
  try
  rethrow
  catch
  block
  try
  br        0
  catch
  local.set 0
  block    i32
  local.get 0
  br_on_exn 0, __cpp_exception
  rethrow
  end_block
  end_try
  end_block
  rethrow
  end_try
  end_try
  end_function


# CHECK:      test_annotation:
# CHECK:        try
# CHECK-NEXT:   br        0               # 0: down to label0
# CHECK-NEXT:   catch                     # catch0:
# CHECK-NEXT:   block
# CHECK-NEXT:   br_if     0               # 0: down to label1
# CHECK-NEXT:   loop                      # label2:
# CHECK-NEXT:   br_if     1               # 1: down to label1
# CHECK-NEXT:   end_loop
# CHECK-NEXT:   end_block                 # label1:
# CHECK-NEXT:   try
# CHECK-NEXT:   rethrow                   # down to catch1
# CHECK-NEXT:   catch                     # catch1:
# CHECK-NEXT:   block
# CHECK-NEXT:   try
# CHECK-NEXT:   br        0               # 0: down to label5
# CHECK-NEXT:   catch                     # catch2:
# CHECK-NEXT:   local.set 0
# CHECK-NEXT:   block    i32
# CHECK-NEXT:   local.get 0
# CHECK-NEXT:   br_on_exn 0, __cpp_exception # 0: down to label6
# CHECK-NEXT:   rethrow                   # to caller
# CHECK-NEXT:   end_block                 # label6:
# CHECK-NEXT:   end_try                   # label5:
# CHECK-NEXT:   end_block                 # label4:
# CHECK-NEXT:   rethrow                   # to caller
# CHECK-NEXT:   end_try                   # label3:
# CHECK-NEXT:   end_try                   # label0:
# CHECK-NEXT:   end_function

