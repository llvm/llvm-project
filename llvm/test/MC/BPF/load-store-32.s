# RUN: llvm-mc -triple bpfel -filetype=obj -o %t %s
# RUN: llvm-objdump --no-print-imm-hex --mattr=+alu32 -d -r %t | FileCheck --check-prefix=CHECK-32 %s
# RUN: llvm-objdump --no-print-imm-hex --mcpu=v1 -d -r %t | FileCheck %s

// ======== BPF_LDX Class ========
  w5 = *(u8 *)(r0 + 0)   // BPF_LDX | BPF_B
  w6 = *(u16 *)(r1 + 8)  // BPF_LDX | BPF_H
  w7 = *(u32 *)(r2 + 16) // BPF_LDX | BPF_W
// CHECK-32: 71 05 00 00 00 00 00 00 	w5 = *(u8 *)(r0 + 0)
// CHECK-32: 69 16 08 00 00 00 00 00 	w6 = *(u16 *)(r1 + 8)
// CHECK-32: 61 27 10 00 00 00 00 00 	w7 = *(u32 *)(r2 + 16)
// CHECK: 71 05 00 00 00 00 00 00 	r5 = *(u8 *)(r0 + 0)
// CHECK: 69 16 08 00 00 00 00 00 	r6 = *(u16 *)(r1 + 8)
// CHECK: 61 27 10 00 00 00 00 00 	r7 = *(u32 *)(r2 + 16)

// ======== BPF_STX Class ========
  *(u8 *)(r0 + 0) = w7    // BPF_STX | BPF_B
  *(u16 *)(r1 + 8) = w8   // BPF_STX | BPF_H
  *(u32 *)(r2 + 16) = w9  // BPF_STX | BPF_W
  lock *(u32 *)(r2 + 16) += w9  // BPF_STX | BPF_W | BPF_XADD
// CHECK-32: 73 70 00 00 00 00 00 00 	*(u8 *)(r0 + 0) = w7
// CHECK-32: 6b 81 08 00 00 00 00 00 	*(u16 *)(r1 + 8) = w8
// CHECK-32: 63 92 10 00 00 00 00 00 	*(u32 *)(r2 + 16) = w9
// CHECK-32: c3 92 10 00 00 00 00 00 	lock *(u32 *)(r2 + 16) += w9
// CHECK: 73 70 00 00 00 00 00 00 	*(u8 *)(r0 + 0) = r7
// CHECK: 6b 81 08 00 00 00 00 00 	*(u16 *)(r1 + 8) = r8
// CHECK: 63 92 10 00 00 00 00 00 	*(u32 *)(r2 + 16) = r9
// CHECK: c3 92 10 00 00 00 00 00 	lock *(u32 *)(r2 + 16) += r9
