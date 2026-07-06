# RUN: llvm-mc -no-type-check -show-encoding -triple=wasm32-unknown-unknown -mattr=+atomics,+relaxed-atomics < %s | FileCheck %s
# RUN: llvm-mc -no-type-check -triple=wasm32-unknown-unknown -mattr=+atomics,+relaxed-atomics %s -filetype=obj -o - | llvm-objdump --mattr=+atomics,+relaxed-atomics --no-print-imm-hex -d - | FileCheck %s --check-prefix=DISASM
# Ensure we can disassemble even when not explicitly enabling the feature in the disassembler.
# In that case we will print nothing instead of "seqcst" but will still print "acqrel"
# RUN: llvm-mc -no-type-check -triple=wasm32-unknown-unknown -mattr=+atomics,+relaxed-atomics %s -filetype=obj -o - | llvm-objdump --no-print-imm-hex -d - | FileCheck %s --check-prefix=DISASM-NOATTR

.section .text.main,"",@
main:
  .functype main () -> ()

  atomic.fence seqcst
  # CHECK: atomic.fence seqcst # encoding: [0xfe,0x03,0x00]
  # DISASM: atomic.fence seqcst
  # DISASM-NOATTR: atomic.fence {{$}}


  atomic.fence acqrel
  # CHECK: atomic.fence acqrel # encoding: [0xfe,0x03,0x01]
  # DISASM: atomic.fence acqrel
  # DISASM-NOATTR: atomic.fence acqrel


  atomic.fence
  # CHECK: atomic.fence seqcst # encoding: [0xfe,0x03,0x00]
  # DISASM: atomic.fence seqcst
  # DISASM-NOATTR: atomic.fence {{$}}


  # CHECK: i32.atomic.load seqcst 0 # encoding: [0xfe,0x10,0x02,0x00]
  # DISASM: i32.atomic.load seqcst 0
  i32.atomic.load 0

  # CHECK: i32.atomic.load seqcst 0 # encoding: [0xfe,0x10,0x02,0x00]
  # DISASM: i32.atomic.load seqcst 0
  i32.atomic.load seqcst 0

  # CHECK: i32.atomic.load acqrel 0 # encoding: [0xfe,0x10,0x22,0x01,0x00]
  # DISASM: i32.atomic.load acqrel 0
  i32.atomic.load acqrel 0

  # CHECK: i64.atomic.load acqrel 0 # encoding: [0xfe,0x11,0x23,0x01,0x00]
  # DISASM: i64.atomic.load acqrel 0
  i64.atomic.load acqrel 0

  # CHECK: i32.atomic.store acqrel 0 # encoding: [0xfe,0x17,0x22,0x01,0x00]
  # DISASM: i32.atomic.store acqrel 0
  i32.atomic.store acqrel 0

  # CHECK: i64.atomic.store acqrel 8 # encoding: [0xfe,0x18,0x23,0x01,0x08]
  # DISASM: i64.atomic.store acqrel 8
  i64.atomic.store acqrel 8

  # CHECK: i32.atomic.rmw.add acqrel 0 # encoding: [0xfe,0x1e,0x22,0x11,0x00]
  # DISASM: i32.atomic.rmw.add acqrel 0
  i32.atomic.rmw.add acqrel 0

  # CHECK: i64.atomic.rmw.cmpxchg acqrel 0 # encoding: [0xfe,0x49,0x23,0x11,0x00]
  # DISASM: i64.atomic.rmw.cmpxchg acqrel 0
  i64.atomic.rmw.cmpxchg acqrel 0

  # CHECK: i32.atomic.load8_u seqcst 0 # encoding: [0xfe,0x12,0x00,0x00]
  # DISASM: i32.atomic.load8_u seqcst 0
  i32.atomic.load8_u seqcst 0:p2align=0

  # CHECK: i64.atomic.rmw32.xchg_u acqrel 0 # encoding: [0xfe,0x47,0x22,0x11,0x00]
  # DISASM: i64.atomic.rmw32.xchg_u acqrel 0
  i64.atomic.rmw32.xchg_u acqrel 0

  end_function
