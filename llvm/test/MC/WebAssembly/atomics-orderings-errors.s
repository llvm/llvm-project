# RUN: not llvm-mc -triple=wasm32-unknown-unknown -mattr=+atomics < %s 2>&1 | FileCheck %s

main:
  .functype main () -> ()

  # CHECK: :[[@LINE+1]]:16: error: memory ordering requires shared-everything feature: acq_rel
  atomic.fence acq_rel

  # CHECK: :[[@LINE+1]]:21: error: memory ordering requires shared-everything feature: acq_rel
  i32.atomic.load 0 acq_rel

  # CHECK: :[[@LINE+1]]:24: error: memory ordering requires shared-everything feature: seq_cst
  i32.atomic.rmw.add 0 seq_cst

  end_function
