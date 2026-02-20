# RUN: llvm-mc -no-type-check -show-encoding -triple=wasm32-unknown-unknown -mattr=+atomics,+shared-everything < %s | FileCheck %s
# RUN: llvm-mc -no-type-check -triple=wasm32-unknown-unknown -mattr=+atomics,+shared-everything %s -filetype=obj -o - | llvm-objdump -d --mattr=+atomics,+shared-everything - | FileCheck %s --check-prefix=DISASM

main:
  .functype main () -> ()

  # CHECK: atomic.fence seq_cst # encoding: [0xfe,0x03,0x04]
  # DISASM: atomic.fence seq_cst
  atomic.fence
  # CHECK: atomic.fence acquire # encoding: [0xfe,0x03,0x01]
  # DISASM: atomic.fence acquire
  atomic.fence acquire
  # CHECK: atomic.fence release # encoding: [0xfe,0x03,0x02]
  # DISASM: atomic.fence release
  atomic.fence release
  # CHECK: atomic.fence acq_rel # encoding: [0xfe,0x03,0x03]
  # DISASM: atomic.fence acq_rel
  atomic.fence acq_rel
  # CHECK: atomic.fence seq_cst # encoding: [0xfe,0x03,0x04]
  # DISASM: atomic.fence seq_cst
  atomic.fence seq_cst

  # CHECK: i32.atomic.load 0 seq_cst # encoding: [0xfe,0x10,0x02,0x00,0x04]
  # DISASM: i32.atomic.load 0 seq_cst
  i32.atomic.load 0
  # CHECK: i32.atomic.load 0 acquire # encoding: [0xfe,0x10,0x02,0x00,0x01]
  # DISASM: i32.atomic.load 0 acquire
  i32.atomic.load 0 acquire
  # CHECK: i32.atomic.load 0 seq_cst # encoding: [0xfe,0x10,0x02,0x00,0x04]
  # DISASM: i32.atomic.load 0 seq_cst
  i32.atomic.load 0 seq_cst

  # CHECK: i64.atomic.load 0 release # encoding: [0xfe,0x11,0x03,0x00,0x02]
  # DISASM: i64.atomic.load 0 release
  i64.atomic.load 0 release

  # CHECK: i32.atomic.store 0 release # encoding: [0xfe,0x17,0x02,0x00,0x02]
  # DISASM: i32.atomic.store 0 release
  i32.atomic.store 0 release

  # CHECK: i64.atomic.store 8 release # encoding: [0xfe,0x18,0x03,0x08,0x02]
  # DISASM: i64.atomic.store 8 release
  i64.atomic.store 8 release

  # CHECK: i32.atomic.rmw.add 0 acq_rel # encoding: [0xfe,0x1e,0x02,0x00,0x03]
  # DISASM: i32.atomic.rmw.add 0 acq_rel
  i32.atomic.rmw.add 0 acq_rel

  # CHECK: i64.atomic.rmw.cmpxchg 0 acquire # encoding: [0xfe,0x49,0x03,0x00,0x01]
  # DISASM: i64.atomic.rmw.cmpxchg 0 acquire
  i64.atomic.rmw.cmpxchg 0 acquire

  # CHECK: i32.atomic.load8_u 0 seq_cst # encoding: [0xfe,0x12,0x00,0x00,0x04]
  # DISASM: i32.atomic.load8_u 0 seq_cst
  i32.atomic.load8_u 0:p2align=0 seq_cst

  # CHECK: i64.atomic.rmw32.xchg_u 0 release # encoding: [0xfe,0x47,0x02,0x00,0x02]
  # DISASM: i64.atomic.rmw32.xchg_u 0 release
  i64.atomic.rmw32.xchg_u 0 release

  end_function
