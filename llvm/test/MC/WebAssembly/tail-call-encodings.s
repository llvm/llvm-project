# RUN: llvm-mc -show-encoding -triple=wasm32-unknown-unknown -mattr=+tail-call < %s | FileCheck %s
# RUN: llvm-mc -show-encoding -triple=wasm32-unknown-unknown -mattr=+reference-types,+tail-call < %s | FileCheck --check-prefix=REF %s
# RUN: llvm-mc --defsym=GC_ENABLED=1 -show-encoding -triple=wasm32-unknown-unknown -mattr=+gc,+reference-types,+tail-call < %s | FileCheck --check-prefix=GC %s

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
    .functype foo2 () -> (i32)

    i32.const 0
    i32.const 0
    # REF: return_call_indirect __indirect_function_table, (i32) -> (i32) # encoding: [0x13,
    # CHECK: return_call_indirect (i32) -> (i32) # encoding: [0x13,
    # CHECK-NEXT: fixup A - offset: 1, value: .Ltypeindex0@TYPEINDEX, kind: fixup_uleb128_i32
    return_call_indirect (i32) -> (i32)

    end_function

.ifdef GC_ENABLED
foo3:
    .functype foo3 () -> (i32)

    i32.const 0
    ref.null_func
    ref.cast (i32) -> (i32)
    # GC: return_call_ref (i32) -> (i32) # encoding: [0x15,
    # GC-NEXT: fixup A - offset: 1, value: .Ltypeindex2@TYPEINDEX, kind: fixup_uleb128_i32
    return_call_ref (i32) -> (i32)

    end_function
.endif
