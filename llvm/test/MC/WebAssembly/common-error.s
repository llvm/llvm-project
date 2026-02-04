# RUN: not llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s 2>&1 | FileCheck %s

# CHECK: error: common symbols are not yet implemented for Wasm: x
# CHECK: error: common symbols are not yet implemented for Wasm: y
        .comm x,4,4
        .comm y,4,4
