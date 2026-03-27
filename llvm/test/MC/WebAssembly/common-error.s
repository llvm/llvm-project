# RUN: not llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s 2>&1 | FileCheck %s

# CHECK:common-error.s:5:9: error: common symbols are not yet implemented for Wasm: x
# CHECK:common-error.s:6:9: error: common symbols are not yet implemented for Wasm: y
        .comm x,4,4
        .comm y,4,4
