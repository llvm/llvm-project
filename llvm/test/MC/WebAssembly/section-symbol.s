# RUN: llvm-mc -triple=wasm32 < %s | FileCheck %s

# check that we can refer to section symbols of other sections.
# getWasmSection currently forces the section symbol to have a suffix.

# TODO: fix the 0-suffix: https://github.com/llvm/llvm-project/issues/48596

    .section    .debug_abbrev,"",@
    .int8       1
    .section    .debug_info,"",@
    .int32      .debug_abbrev0

# CHECK:         .section    .debug_abbrev,"",@
# CHECK-NEXT:    .int8       1
# CHECK-NEXT:    .section    .debug_info,"",@
# CHECK-NEXT:    .int32      .debug_abbrev0
