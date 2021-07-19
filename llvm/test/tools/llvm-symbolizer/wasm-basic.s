# REQUIRES: webassembly-registered-target
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s -o %t.o -g

foo:
    .functype foo () -> ()
    nop
    end_function

bar:
    .functype bar (i32) -> (i32)
    local.get 0
    return
    end_function

# RUN: llvm-symbolizer -e %t.o 3 4 7 8 | FileCheck %s
## Byte 1 is the function length and 2 is the locals declaration.
## Currently no line corresponds to them.
## TODO: create a loc for .functype?

## Test 2 functions to ensure wasm's function-sections system works.
# CHECK: wasm-basic.s:6:0
# CHECK: wasm-basic.s:7:0
# CHECK: wasm-basic.s:11:0
# CHECK: wasm-basic.s:11:0
