# RUN: llvm-mc -no-type-check -show-encoding -triple=wasm32-unknown-unknown -mattr=+atomics,+shared-everything < %s | FileCheck %s
# RUN: llvm-mc -no-type-check -triple=wasm32-unknown-unknown -mattr=+atomics,+shared-everything %s -filetype=obj -o - | llvm-objdump -d --mattr=+atomics,+shared-everything - | FileCheck %s --check-prefix=DISASM

main:
  .functype main () -> ()

  # CHECK: atomic.fence seq_cst # encoding: [0xfe,0x03,0x00]
  # DISASM: atomic.fence seq_cst
  atomic.fence
  # CHECK: atomic.fence acq_rel # encoding: [0xfe,0x03,0x01]
  # DISASM: atomic.fence acq_rel
  atomic.fence acq_rel
  # CHECK: atomic.fence seq_cst # encoding: [0xfe,0x03,0x00]
  # DISASM: atomic.fence seq_cst
  atomic.fence seq_cst

  # CHECK: i32.atomic.load 0 seq_cst # encoding: [0xfe,0x10,0x02,0x00]
  # DISASM: i32.atomic.load 0 seq_cst
  i32.atomic.load 0
  # CHECK: i32.atomic.load 0 acq_rel # encoding: [0xfe,0x10,0x22,0x00,0x01]
  # DISASM: i32.atomic.load 0 acq_rel
  i32.atomic.load 0 acq_rel
  # CHECK: i32.atomic.load 0 seq_cst # encoding: [0xfe,0x10,0x02,0x00]
  # DISASM: i32.atomic.load 0 seq_cst
  i32.atomic.load 0 seq_cst

  # CHECK: i64.atomic.load 0 acq_rel # encoding: [0xfe,0x11,0x23,0x00,0x01]
  # DISASM: i64.atomic.load 0 acq_rel
  i64.atomic.load 0 acq_rel

  # CHECK: i32.atomic.store 0 acq_rel # encoding: [0xfe,0x17,0x22,0x00,0x01]
  # DISASM: i32.atomic.store 0 acq_rel
  i32.atomic.store 0 acq_rel

  # CHECK: i64.atomic.store 8 acq_rel # encoding: [0xfe,0x18,0x23,0x08,0x01]
  # DISASM: i64.atomic.store 8 acq_rel
  i64.atomic.store 8 acq_rel

  # CHECK: i32.atomic.rmw.add 0 acq_rel # encoding: [0xfe,0x1e,0x22,0x00,0x11]
  # DISASM: i32.atomic.rmw.add 0 acq_rel
  i32.atomic.rmw.add 0 acq_rel

  # CHECK: i64.atomic.rmw.cmpxchg 0 acq_rel # encoding: [0xfe,0x49,0x23,0x00,0x11]
  # DISASM: i64.atomic.rmw.cmpxchg 0 acq_rel
  i64.atomic.rmw.cmpxchg 0 acq_rel

  # CHECK: i32.atomic.load8_u 0 seq_cst # encoding: [0xfe,0x12,0x00,0x00]
  # DISASM: i32.atomic.load8_u 0 seq_cst
  i32.atomic.load8_u 0:p2align=0 seq_cst

  # CHECK: i64.atomic.rmw32.xchg_u 0 acq_rel # encoding: [0xfe,0x47,0x22,0x00,0x11]
  # DISASM: i64.atomic.rmw32.xchg_u 0 acq_rel
  i64.atomic.rmw32.xchg_u 0 acq_rel

  end_function
