; RUN: split-file %s %t
; RUN: not llvm-as < %t/aarch64-svcount.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-AARCH64-SVCOUNT %s
; RUN: not llvm-as < %t/riscv-vector-tuple.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-RISCV-VECTOR-TUPLE %s
; RUN: not llvm-as < %t/amdgcn-named-barrier.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-AMDGCN-NAMEDBARRIER %s
; RUN: not llvm-as < %t/amdgcn-stridemark-type-param.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-AMDGCN-STRIDEMARK-TYPE %s
; RUN: not llvm-as < %t/amdgcn-stridemark-int-params.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-AMDGCN-STRIDEMARK-INTS %s
; Check target extension type properties are verified in the assembler.

;--- aarch64-svcount.ll
declare target("aarch64.svcount", i32) @aarch64_svcount()
; CHECK-AARCH64-SVCOUNT: error: target extension type aarch64.svcount should have no parameters

;--- riscv-vector-tuple.ll
declare target("riscv.vector.tuple", 99) @riscv_vector_tuple()
; CHECK-RISCV-VECTOR-TUPLE: target extension type riscv.vector.tuple should have one type parameter and one integer parameter

;--- amdgcn-named-barrier.ll
declare target("amdgcn.named.barrier", i32) @amdgcn_named_barrier()
; CHECK-AMDGCN-NAMEDBARRIER: target extension type amdgcn.named.barrier should have no type parameters and one integer parameter

;--- amdgcn-stridemark-type-param.ll
declare target("amdgcn.stridemark", i32) @amdgcn_stridemark_type_param()
; CHECK-AMDGCN-STRIDEMARK-TYPE: target extension type amdgcn.stridemark should have no type parameters and at most one integer parameter

;--- amdgcn-stridemark-int-params.ll
declare target("amdgcn.stridemark", 16, 32) @amdgcn_stridemark_int_params()
; CHECK-AMDGCN-STRIDEMARK-INTS: target extension type amdgcn.stridemark should have no type parameters and at most one integer parameter
