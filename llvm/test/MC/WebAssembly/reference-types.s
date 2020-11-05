# RUN: llvm-mc -show-encoding -triple=wasm32-unknown-unknown -mattr=+reference-types < %s | FileCheck %s
# RUN: llvm-mc -show-encoding -triple=wasm64-unknown-unknown -mattr=+reference-types < %s | FileCheck %s

#      CHECK: ref_null_externref:
# CHECK-NEXT:         .functype	ref_null_externref () -> (externref)
#      CHECK:	ref.null extern # encoding: [0xd0,0x6f]
# CHECK-NEXT:	end_function
ref_null_externref:
  .functype ref_null_externref () -> (externref)
  ref.null extern
  end_function

#      CHECK: ref_null_funcref:
# CHECK-NEXT:         .functype	ref_null_funcref () -> (funcref)
#      CHECK:	ref.null func # encoding: [0xd0,0x70]
# CHECK-NEXT:	end_function
ref_null_funcref:
  .functype ref_null_funcref () -> (funcref)
  ref.null func
  end_function
