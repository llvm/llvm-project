# RUN: not llvm-mc -triple=wasm32-unknown-unknown -mattr=+atomics < %s 2>&1 | FileCheck %s

main:
  .functype main () -> ()

  # CHECK: :[[@LINE+1]]:16: error: memory ordering requires relaxed-atomics feature: acqrel
  atomic.fence acqrel

  # CHECK: :[[@LINE+1]]:19: error: memory ordering requires relaxed-atomics feature: acqrel
  i32.atomic.load acqrel 0

  # CHECK: :[[@LINE+1]]:22: error: memory ordering requires relaxed-atomics feature: seqcst
  i32.atomic.rmw.add seqcst 0

  end_function
