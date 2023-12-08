# RUN: split-file %s %t
# RUN: llvm-mc -show-encoding -triple=wasm32-unknown-unknown -mattr=+bulk-memory < %t/wasm32.s | FileCheck %s
# RUN: llvm-mc -show-encoding -triple=wasm64-unknown-unknown -mattr=+bulk-memory < %t/wasm64.s | FileCheck %s

#--- wasm32.s
main:
    .functype main () -> ()

    i32.const 2 # dest address
    i32.const 3 # src offset
    i32.const 4 # count
    # CHECK: memory.init 3, 0 # encoding: [0xfc,0x08,0x03,0x00]
    memory.init 3, 0

    # CHECK: data.drop 3 # encoding: [0xfc,0x09,0x03]
    data.drop 3

    i32.const 2 # dst
    i32.const 3 # src
    i32.const 4 # count
    # CHECK: memory.copy 0, 0 # encoding: [0xfc,0x0a,0x00,0x00]
    memory.copy 0, 0

    i32.const 2 # addr
    i32.const 3 # val
    i32.const 4 # count
    # CHECK: memory.fill 0 # encoding: [0xfc,0x0b,0x00]
    memory.fill 0

    end_function

#--- wasm64.s
main:
    .functype main () -> ()

    i64.const 2 # dest address
    i32.const 3 # src offset
    i32.const 4 # count
    memory.init 3, 0

    data.drop 3

    i64.const 2 # dst
    i64.const 3 # src
    i64.const 4 # count
    memory.copy 0, 0

    i64.const 2 # addr
    i32.const 3 # val
    i64.const 4 # count
    memory.fill 0

    end_function
