# RUN: not llvm-mc -triple=wasm32-unknown-unknown -mattr=+atomics < %s 2>&1 | FileCheck %s

main:
  .functype main () -> ()

  # CHECK: :[[@LINE+1]]:16: error: memory ordering requires shared-everything feature: acqrel
  atomic.fence acqrel

  # CHECK: :[[@LINE+1]]:21: error: memory ordering requires shared-everything feature: acqrel
  i32.atomic.load 0 acqrel

  # CHECK: :[[@LINE+1]]:24: error: memory ordering requires shared-everything feature: seqcst
  i32.atomic.rmw.add 0 seqcst

  end_function
