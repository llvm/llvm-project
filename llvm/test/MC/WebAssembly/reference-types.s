# RUN: llvm-mc -show-encoding -triple=wasm32-unknown-unknown -mattr=+reference-types < %s | FileCheck %s
# RUN: llvm-mc -show-encoding -triple=wasm64-unknown-unknown -mattr=+reference-types < %s | FileCheck %s

# CHECK-LABEL: ref_null_test:
# CHECK: ref.null func   # encoding: [0xd0,0x70]
# CHECK: ref.null extern # encoding: [0xd0,0x6f]
ref_null_test:
  .functype ref_null_test () -> ()
  ref.null func
  drop
  ref.null extern
  drop
  end_function

# CHECK-LABEL: ref_sig_test_funcref:
# CHECK-NEXT: .functype ref_sig_test_funcref (funcref) -> (funcref)
ref_sig_test_funcref:
  .functype ref_sig_test_funcref (funcref) -> (funcref)
  end_function

# CHECK-LABEL: ref_sig_test_externref:
# CHECK-NEXT: .functype ref_sig_test_externref (externref) -> (externref)
ref_sig_test_externref:
  .functype ref_sig_test_externref (externref) -> (externref)
  end_function

# CHECK-LABEL: ref_select_test:
# CHECK: funcref.select   # encoding: [0x1b]
# CHECK: externref.select # encoding: [0x1b]
ref_select_test:
  .functype ref_select_test () -> ()
  ref.null func
  ref.null func
  i32.const 0
  funcref.select
  drop
  ref.null extern
  ref.null extern
  i32.const 0
  externref.select
  drop
  end_function

# CHECK-LABEL: ref_block_test:
# CHECK: block funcref
# CHECK: block externref
ref_block_test:
  .functype ref_block_test () -> ()
  block funcref
  block externref
  end_block
  end_block
  end_function
