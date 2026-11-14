# RUN: llvm-mc -triple bpfel -filetype=obj -o %t %s
# RUN: llvm-objdump --no-print-imm-hex -d -r %t | FileCheck %s

# r12 points to the frame of the function in the stack in arena.

  r12 += 16
// CHECK: 07 0c 00 00 10 00 00 00 	r12 += 16
  r1 = r12
// CHECK: bf c1 00 00 00 00 00 00 	r1 = r12
  r2 = *(u64 *)(r12 - 8)
// CHECK: 79 c2 f8 ff 00 00 00 00 	r2 = *(u64 *)(r12 - 8)
  w2 = *(u8 *)(r12 - 1)
// CHECK: 71 c2 ff ff 00 00 00 00 	w2 = *(u8 *)(r12 - 1)
  r2 = *(s32 *)(r12 - 4)
// CHECK: 81 c2 fc ff 00 00 00 00 	r2 = *(s32 *)(r12 - 4)
  *(u64 *)(r12 - 8) = r2
// CHECK: 7b 2c f8 ff 00 00 00 00 	*(u64 *)(r12 - 8) = r2
  *(u16 *)(r12 - 2) = w2
// CHECK: 6b 2c fe ff 00 00 00 00 	*(u16 *)(r12 - 2) = w2
  *(u32 *)(r12 - 16) = 7
// CHECK: 62 0c f0 ff 07 00 00 00 	*(u32 *)(r12 - 16) = 7
  w12 += 16
// CHECK: 04 0c 00 00 10 00 00 00 	w12 += 16
  w1 = w12
// CHECK: bc c1 00 00 00 00 00 00 	w1 = w12

# There is no r13.
  .byte 0xbf, 0xd1, 0, 0, 0, 0, 0, 0
// CHECK: bf d1 00 00 00 00 00 00 	<unknown>
  .byte 0xbf, 0x1d, 0, 0, 0, 0, 0, 0
// CHECK: bf 1d 00 00 00 00 00 00 	<unknown>
  .byte 0x79, 0xd2, 0xf8, 0xff, 0, 0, 0, 0
// CHECK: 79 d2 f8 ff 00 00 00 00 	<unknown>
