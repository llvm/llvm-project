# RUN: llvm-mc -show-encoding -triple=wasm32-unkown-unknown -mattr=+tail-call < %s | FileCheck %s

bar1:
    .functype bar1 () -> ()
    end_function

foo1:
    .functype foo1 () -> ()

    # CHECK: return_call bar1  # encoding: [0x12,
    # CHECK-NEXT: fixup A - offset: 1, value: bar1, kind: fixup_uleb128_i32
    return_call bar1

    end_function

foo2:
    .functype foo2 () -> ()

    # CHECK: return_call_indirect 0 # encoding: [0x13,0x00,0x00]
    return_call_indirect 0

    end_function
