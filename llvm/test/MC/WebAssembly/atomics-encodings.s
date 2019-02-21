# RUN: llvm-mc -show-encoding -triple=wasm32-unkown-unknown -mattr=+atomics < %s | FileCheck %s

main:
  .functype main () -> ()

  # FIXME This doesn't work because of PR40728. Enable this once it's fixed.
  # C HECK:  atomic.notify 0:p2align=0 # encoding: [0xfe,0x00,0x00,0x00]
  # atomic.notify 0
  # CHECK:  i32.atomic.wait 0:p2align=0 # encoding: [0xfe,0x01,0x00,0x00]
  i32.atomic.wait 0
  # CHECK:  i64.atomic.wait 0:p2align=0 # encoding: [0xfe,0x02,0x00,0x00]
  i64.atomic.wait 0

  # CHECK: i32.atomic.load 0:p2align=0 # encoding: [0xfe,0x10,0x00,0x00]
  i32.atomic.load 0
  # CHECK: i64.atomic.load 4:p2align=0 # encoding: [0xfe,0x11,0x00,0x04]
  i64.atomic.load 4
  # CHECK:  i32.atomic.load8_u 48 # encoding: [0xfe,0x12,0x00,0x30]
  i32.atomic.load8_u 48
  # CHECK:  i32.atomic.load16_u 0:p2align=0 # encoding: [0xfe,0x13,0x00,0x00]
  i32.atomic.load16_u 0
  # CHECK:  i64.atomic.load8_u 0 # encoding: [0xfe,0x14,0x00,0x00]
  i64.atomic.load8_u 0
  # CHECK:  i64.atomic.load16_u 0:p2align=0 # encoding: [0xfe,0x15,0x00,0x00]
  i64.atomic.load16_u 0
  # CHECK:  i64.atomic.load32_u 0:p2align=0 # encoding: [0xfe,0x16,0x00,0x00]
  i64.atomic.load32_u 0

  # CHECK:  i32.atomic.store 0:p2align=0 # encoding: [0xfe,0x17,0x00,0x00]
  i32.atomic.store 0
  # CHECK:  i64.atomic.store 8:p2align=0 # encoding: [0xfe,0x18,0x00,0x08]
  i64.atomic.store 8
  # CHECK:  i32.atomic.store8 0 # encoding: [0xfe,0x19,0x00,0x00]
  i32.atomic.store8 0
  # CHECK:  i32.atomic.store16 0:p2align=0 # encoding: [0xfe,0x1a,0x00,0x00]
  i32.atomic.store16 0
  # CHECK:  i64.atomic.store8 16 # encoding: [0xfe,0x1b,0x00,0x10]
  i64.atomic.store8 16
  # CHECK:  i64.atomic.store16 0:p2align=0 # encoding: [0xfe,0x1c,0x00,0x00]
  i64.atomic.store16 0
  # CHECK:  i64.atomic.store32 0:p2align=0 # encoding: [0xfe,0x1d,0x00,0x00]
  i64.atomic.store32 0

  # CHECK:  i32.atomic.rmw.add 0:p2align=0 # encoding: [0xfe,0x1e,0x00,0x00]
  i32.atomic.rmw.add 0
  # CHECK:  i64.atomic.rmw.add 0:p2align=0 # encoding: [0xfe,0x1f,0x00,0x00]
  i64.atomic.rmw.add 0
  # CHECK:  i32.atomic.rmw8.add_u 0 # encoding: [0xfe,0x20,0x00,0x00]
  i32.atomic.rmw8.add_u 0
  # CHECK:  i32.atomic.rmw16.add_u 0:p2align=0 # encoding: [0xfe,0x21,0x00,0x00]
  i32.atomic.rmw16.add_u 0
  # CHECK:  i64.atomic.rmw8.add_u 0 # encoding: [0xfe,0x22,0x00,0x00]
  i64.atomic.rmw8.add_u 0
  # CHECK:  i64.atomic.rmw16.add_u 0:p2align=0 # encoding: [0xfe,0x23,0x00,0x00]
  i64.atomic.rmw16.add_u 0
  # CHECK:  i64.atomic.rmw32.add_u 16:p2align=0 # encoding: [0xfe,0x24,0x00,0x10]
  i64.atomic.rmw32.add_u 16

  # CHECK:  i32.atomic.rmw.sub 0:p2align=0 # encoding: [0xfe,0x25,0x00,0x00]
  i32.atomic.rmw.sub 0
  # CHECK:  i64.atomic.rmw.sub 0:p2align=0 # encoding: [0xfe,0x26,0x00,0x00]
  i64.atomic.rmw.sub 0
  # CHECK:  i32.atomic.rmw8.sub_u 0 # encoding: [0xfe,0x27,0x00,0x00]
  i32.atomic.rmw8.sub_u 0
  # CHECK:  i32.atomic.rmw16.sub_u 0:p2align=0 # encoding: [0xfe,0x28,0x00,0x00]
  i32.atomic.rmw16.sub_u 0
  # CHECK:  i64.atomic.rmw8.sub_u 8 # encoding: [0xfe,0x29,0x00,0x08]
  i64.atomic.rmw8.sub_u 8
  # CHECK:  i64.atomic.rmw16.sub_u 0:p2align=0 # encoding: [0xfe,0x2a,0x00,0x00]
  i64.atomic.rmw16.sub_u 0
  # CHECK:  i64.atomic.rmw32.sub_u 0:p2align=0 # encoding: [0xfe,0x2b,0x00,0x00]
  i64.atomic.rmw32.sub_u 0

  # CHECK:  i32.atomic.rmw.and 0:p2align=0 # encoding: [0xfe,0x2c,0x00,0x00]
  i32.atomic.rmw.and 0
  # CHECK:  i64.atomic.rmw.and 0:p2align=0 # encoding: [0xfe,0x2d,0x00,0x00]
  i64.atomic.rmw.and 0
  # CHECK:  i32.atomic.rmw8.and_u 0 # encoding: [0xfe,0x2e,0x00,0x00]
  i32.atomic.rmw8.and_u 0
  # CHECK:  i32.atomic.rmw16.and_u 0:p2align=0 # encoding: [0xfe,0x2f,0x00,0x00]
  i32.atomic.rmw16.and_u 0
  # CHECK:  i64.atomic.rmw8.and_u 96 # encoding: [0xfe,0x30,0x00,0x60]
  i64.atomic.rmw8.and_u 96
  # CHECK:  i64.atomic.rmw16.and_u 0:p2align=0 # encoding: [0xfe,0x31,0x00,0x00]
  i64.atomic.rmw16.and_u 0
  # CHECK:  i64.atomic.rmw32.and_u 0:p2align=0 # encoding: [0xfe,0x32,0x00,0x00]
  i64.atomic.rmw32.and_u 0

  # CHECK:  i32.atomic.rmw.or 0:p2align=0 # encoding: [0xfe,0x33,0x00,0x00]
  i32.atomic.rmw.or 0
  # CHECK:  i64.atomic.rmw.or 0:p2align=0 # encoding: [0xfe,0x34,0x00,0x00]
  i64.atomic.rmw.or 0
  # CHECK:  i32.atomic.rmw8.or_u 0 # encoding: [0xfe,0x35,0x00,0x00]
  i32.atomic.rmw8.or_u 0
  # CHECK:  i32.atomic.rmw16.or_u 0:p2align=0 # encoding: [0xfe,0x36,0x00,0x00]
  i32.atomic.rmw16.or_u 0
  # CHECK:  i64.atomic.rmw8.or_u 0 # encoding: [0xfe,0x37,0x00,0x00]
  i64.atomic.rmw8.or_u 0
  # CHECK:  i64.atomic.rmw16.or_u 48:p2align=0 # encoding: [0xfe,0x38,0x00,0x30]
  i64.atomic.rmw16.or_u 48
  # CHECK:  i64.atomic.rmw32.or_u 0:p2align=0 # encoding: [0xfe,0x39,0x00,0x00]
  i64.atomic.rmw32.or_u 0

  # CHECK:  i32.atomic.rmw.xor 0:p2align=0 # encoding: [0xfe,0x3a,0x00,0x00]
  i32.atomic.rmw.xor 0
  # CHECK:  i64.atomic.rmw.xor 0:p2align=0 # encoding: [0xfe,0x3b,0x00,0x00]
  i64.atomic.rmw.xor 0
  # CHECK:  i32.atomic.rmw8.xor_u 4 # encoding: [0xfe,0x3c,0x00,0x04]
  i32.atomic.rmw8.xor_u 4
  # CHECK:  i32.atomic.rmw16.xor_u 0:p2align=0 # encoding: [0xfe,0x3d,0x00,0x00]
  i32.atomic.rmw16.xor_u 0
  # CHECK:  i64.atomic.rmw8.xor_u 0 # encoding: [0xfe,0x3e,0x00,0x00]
  i64.atomic.rmw8.xor_u 0
  # CHECK:  i64.atomic.rmw16.xor_u 0:p2align=0 # encoding: [0xfe,0x3f,0x00,0x00]
  i64.atomic.rmw16.xor_u 0
  # CHECK:  i64.atomic.rmw32.xor_u 0:p2align=0 # encoding: [0xfe,0x40,0x00,0x00]
  i64.atomic.rmw32.xor_u 0

  # CHECK:  i32.atomic.rmw.xchg 0:p2align=0 # encoding: [0xfe,0x41,0x00,0x00]
  i32.atomic.rmw.xchg 0
  # CHECK:  i64.atomic.rmw.xchg 0:p2align=0 # encoding: [0xfe,0x42,0x00,0x00]
  i64.atomic.rmw.xchg 0
  # CHECK:  i32.atomic.rmw8.xchg_u 0 # encoding: [0xfe,0x43,0x00,0x00]
  i32.atomic.rmw8.xchg_u 0
  # CHECK:  i32.atomic.rmw16.xchg_u 0:p2align=0 # encoding: [0xfe,0x44,0x00,0x00]
  i32.atomic.rmw16.xchg_u 0
  # CHECK:  i64.atomic.rmw8.xchg_u 0 # encoding: [0xfe,0x45,0x00,0x00]
  i64.atomic.rmw8.xchg_u 0
  # CHECK:  i64.atomic.rmw16.xchg_u 8:p2align=0 # encoding: [0xfe,0x46,0x00,0x08]
  i64.atomic.rmw16.xchg_u 8
  # CHECK:  i64.atomic.rmw32.xchg_u 0:p2align=0 # encoding: [0xfe,0x47,0x00,0x00]
  i64.atomic.rmw32.xchg_u 0

  # CHECK:  i32.atomic.rmw.cmpxchg 32:p2align=0 # encoding: [0xfe,0x48,0x00,0x20]
  i32.atomic.rmw.cmpxchg 32
  # CHECK:  i64.atomic.rmw.cmpxchg 0:p2align=0 # encoding: [0xfe,0x49,0x00,0x00]
  i64.atomic.rmw.cmpxchg 0
  # CHECK:  i32.atomic.rmw8.cmpxchg_u 0 # encoding: [0xfe,0x4a,0x00,0x00]
  i32.atomic.rmw8.cmpxchg_u 0
  # CHECK:  i32.atomic.rmw16.cmpxchg_u 0:p2align=0 # encoding: [0xfe,0x4b,0x00,0x00]
  i32.atomic.rmw16.cmpxchg_u 0
  # CHECK:  i64.atomic.rmw8.cmpxchg_u 16 # encoding: [0xfe,0x4c,0x00,0x10]
  i64.atomic.rmw8.cmpxchg_u 16
  # CHECK:  i64.atomic.rmw16.cmpxchg_u 0:p2align=0 # encoding: [0xfe,0x4d,0x00,0x00]
  i64.atomic.rmw16.cmpxchg_u 0
  # CHECK:  i64.atomic.rmw32.cmpxchg_u 0:p2align=0 # encoding: [0xfe,0x4e,0x00,0x00]
  i64.atomic.rmw32.cmpxchg_u 0

  end_function
