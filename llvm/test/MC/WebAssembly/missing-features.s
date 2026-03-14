# RUN: not llvm-mc -triple=wasm32-unknown-unknown  < %s 2>&1 | FileCheck %s

# Check that missing features are named in the error message

# CHECK: error: instruction requires: simd128
needs_simd:
    .functype needs_simd () -> (v128)
    i32.const 42
    i32x4.splat
    drop
    end_function
