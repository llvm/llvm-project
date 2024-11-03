; RUN: split-file %s %t
; RUN: not llvm-as < %t/aarch64-svcount.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-AARCH64-SVCOUNT %s
; RUN: not llvm-as < %t/riscv-vector-tuple.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-RISCV-VECTOR-TUPLE %s
; Check target extension type properties are verified in the assembler.

;--- aarch64-svcount.ll
declare target("aarch64.svcount", i32) @aarch64_svcount()
; CHECK-AARCH64-SVCOUNT: error: target extension type aarch64.svcount should have no parameters

;--- riscv-vector-tuple.ll
declare target("riscv.vector.tuple", 99) @riscv_vector_tuple()
; CHECK-RISCV-VECTOR-TUPLE: target extension type riscv.vector.tuple should have one type parameter and one integer parameter
