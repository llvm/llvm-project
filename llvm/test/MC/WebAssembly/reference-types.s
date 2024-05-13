# RUN: llvm-mc -show-encoding -triple=wasm32-unknown-unknown -mattr=+reference-types < %s | FileCheck %s
# RUN: llvm-mc -show-encoding -triple=wasm64-unknown-unknown -mattr=+reference-types < %s | FileCheck %s

# CHECK-LABEL:ref_is_null:
# CHECK: ref.is_null     # encoding: [0xd1]
ref_is_null:
  .functype ref_is_null () -> (i32, i32)
  ref.null_extern
  ref.is_null
  ref.null_func
  ref.is_null
  end_function

# CHECK-LABEL: ref_null_test:
# CHECK: ref.null_func   # encoding: [0xd0,0x70]
# CHECK: ref.null_extern # encoding: [0xd0,0x6f]
ref_null_test:
  .functype ref_null_test () -> ()
  ref.null_func
  drop
  ref.null_extern
  drop
  end_function

# CHECK-LABEL: ref_sig_test_funcref:
# CHECK-NEXT: .functype ref_sig_test_funcref (funcref) -> (funcref)
ref_sig_test_funcref:
  .functype ref_sig_test_funcref (funcref) -> (funcref)
  local.get 0
  end_function

# CHECK-LABEL: ref_sig_test_externref:
# CHECK-NEXT: .functype ref_sig_test_externref (externref) -> (externref)
ref_sig_test_externref:
  .functype ref_sig_test_externref (externref) -> (externref)
  local.get 0
  end_function

# CHECK-LABEL: ref_select_test:
# CHECK: funcref.select   # encoding: [0x1b]
# CHECK: externref.select # encoding: [0x1b]
ref_select_test:
  .functype ref_select_test () -> ()
  ref.null_func
  ref.null_func
  i32.const 0
  funcref.select
  drop
  ref.null_extern
  ref.null_extern
  i32.const 0
  externref.select
  drop
  end_function

# CHECK-LABEL: ref_block_test:
# CHECK: block funcref
# CHECK: block externref
ref_block_test:
  .functype ref_block_test () -> (externref, funcref)
  block funcref
  block externref
  ref.null_extern
  end_block
  ref.null_func
  end_block
  end_function
