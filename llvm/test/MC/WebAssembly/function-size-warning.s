# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s -o %t.o 2>&1 | FileCheck %s
# RUN: llvm-objdump -t %t.o

foo:
  .functype foo () -> ()
  i32.const 1
  drop
  end_function

# .size directives for functions are no longer required and will
# be ignored but we continue to allow them to support legacy
# assembly files.
.size foo, 0

# CHECK: warning: .size directive ignored for function symbols
