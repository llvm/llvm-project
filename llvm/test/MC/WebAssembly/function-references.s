# RUN: llvm-mc -show-encoding -triple=wasm32-unknown-unknown -mattr=+reference-types,+tail-call < %s | FileCheck %s
# RUN: llvm-mc -show-encoding -triple=wasm64-unknown-unknown -mattr=+reference-types,+tail-call < %s | FileCheck %s

# Verify that dropping the tail-call feature rejects return_call_ref. The
# reference-types predicate is not enforced at the parser level today (shared
# with other Requires<[HasReferenceTypes]> opcodes such as ref.null_func), so
# we only assert the tail-call gate.
# RUN: not llvm-mc -triple=wasm32-unknown-unknown -mattr=+reference-types %s 2>&1 \
# RUN:   | FileCheck --check-prefix=NO-TAIL-CALL %s

# NO-TAIL-CALL: error: instruction requires: tail-call

call_ref_void:
    .functype call_ref_void () -> ()
    ref.null_func
    # CHECK: call_ref     () -> () # encoding: [0x14,
    # CHECK-NEXT: fixup A - offset: 1, value: .Ltypeindex0@TYPEINDEX, kind: fixup_uleb128_i32
    call_ref () -> ()
    end_function

call_ref_sig:
    .functype call_ref_sig () -> (i32)
    i32.const 1
    i32.const 2
    ref.null_func
    # CHECK: call_ref     (i32, i32) -> (i32) # encoding: [0x14,
    # CHECK-NEXT: fixup A - offset: 1, value: .Ltypeindex1@TYPEINDEX, kind: fixup_uleb128_i32
    call_ref (i32, i32) -> (i32)
    end_function

return_call_ref_void:
    .functype return_call_ref_void () -> ()
    ref.null_func
    # CHECK: return_call_ref     () -> () # encoding: [0x15,
    # CHECK-NEXT: fixup A - offset: 1, value: .Ltypeindex2@TYPEINDEX, kind: fixup_uleb128_i32
    return_call_ref () -> ()
    end_function
