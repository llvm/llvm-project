# RUN: llvm-mc -no-type-check -show-encoding -triple=wasm32-unknown-unknown -mattr=+atomics,+shared-everything < %s | FileCheck %s
# RUN: llvm-mc -no-type-check -triple=wasm32-unknown-unknown -mattr=+atomics,+shared-everything %s -filetype=obj -o - | llvm-objdump -d --mattr=+atomics,+shared-everything - | FileCheck %s --check-prefix=DISASM

main:
  .functype main () -> ()

  # CHECK: atomic.fence seqcst # encoding: [0xfe,0x03,0x00]
  # DISASM: atomic.fence seqcst
  atomic.fence
  # CHECK: atomic.fence acqrel # encoding: [0xfe,0x03,0x01]
  # DISASM: atomic.fence acqrel
  atomic.fence acqrel
  # CHECK: atomic.fence seqcst # encoding: [0xfe,0x03,0x00]
  # DISASM: atomic.fence seqcst
  atomic.fence seqcst

  # CHECK: i32.atomic.load 0 seqcst # encoding: [0xfe,0x10,0x02,0x00]
  # DISASM: i32.atomic.load 0 seqcst
  i32.atomic.load 0
  # CHECK: i32.atomic.load 0 acqrel # encoding: [0xfe,0x10,0x22,0x00,0x01]
  # DISASM: i32.atomic.load 0 acqrel
  i32.atomic.load 0 acqrel
  # CHECK: i32.atomic.load 0 seqcst # encoding: [0xfe,0x10,0x02,0x00]
  # DISASM: i32.atomic.load 0 seqcst
  i32.atomic.load 0 seqcst

  # CHECK: i64.atomic.load 0 acqrel # encoding: [0xfe,0x11,0x23,0x00,0x01]
  # DISASM: i64.atomic.load 0 acqrel
  i64.atomic.load 0 acqrel

  # CHECK: i32.atomic.store 0 acqrel # encoding: [0xfe,0x17,0x22,0x00,0x01]
  # DISASM: i32.atomic.store 0 acqrel
  i32.atomic.store 0 acqrel

  # CHECK: i64.atomic.store 8 acqrel # encoding: [0xfe,0x18,0x23,0x08,0x01]
  # DISASM: i64.atomic.store 8 acqrel
  i64.atomic.store 8 acqrel

  # CHECK: i32.atomic.rmw.add 0 acqrel # encoding: [0xfe,0x1e,0x22,0x00,0x11]
  # DISASM: i32.atomic.rmw.add 0 acqrel
  i32.atomic.rmw.add 0 acqrel

  # CHECK: i64.atomic.rmw.cmpxchg 0 acqrel # encoding: [0xfe,0x49,0x23,0x00,0x11]
  # DISASM: i64.atomic.rmw.cmpxchg 0 acqrel
  i64.atomic.rmw.cmpxchg 0 acqrel

  # CHECK: i32.atomic.load8_u 0 seqcst # encoding: [0xfe,0x12,0x00,0x00]
  # DISASM: i32.atomic.load8_u 0 seqcst
  i32.atomic.load8_u 0:p2align=0 seqcst

  # CHECK: i64.atomic.rmw32.xchg_u 0 acqrel # encoding: [0xfe,0x47,0x22,0x00,0x11]
  # DISASM: i64.atomic.rmw32.xchg_u 0 acqrel
  i64.atomic.rmw32.xchg_u 0 acqrel

  end_function
