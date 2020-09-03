# RUN: llvm-mc -show-encoding -triple=wasm32-unknown-unknown -mattr=+bulk-memory < %s | FileCheck %s
# RUN: llvm-mc -show-encoding -triple=wasm64-unknown-unknown -mattr=+bulk-memory < %s | FileCheck %s

main:
    .functype main () -> ()

    # CHECK: memory.init 3, 0 # encoding: [0xfc,0x08,0x03,0x00]
    memory.init 3, 0

    # CHECK: data.drop 3 # encoding: [0xfc,0x09,0x03]
    data.drop 3

    # CHECK: memory.copy 0, 0 # encoding: [0xfc,0x0a,0x00,0x00]
    memory.copy 0, 0

    # CHECK: memory.fill 0 # encoding: [0xfc,0x0b,0x00]
    memory.fill 0

    end_function
