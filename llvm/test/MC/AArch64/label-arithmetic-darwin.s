// RUN: llvm-mc -triple aarch64-darwin -filetype=obj %s -o - | llvm-objdump -r -d - | FileCheck %s
// RUN: llvm-mc -triple aarch64-ios -filetype=obj %s -o - | llvm-objdump -r -d - | FileCheck %s

visible:
  .space 8
Lstart:
  .space 8
Lend:
  adds w0, w1, #(Lend - Lstart)
  adds x0, x1, #(Lend - Lstart)
  add w0, w1, #(Lend - Lstart)
  add x0, x1, #(Lend - Lstart)
  cmp w0, #(Lend - Lstart)
  cmp x0, #(Lend - Lstart)
  sub w0, w1, #(Lend - Lstart)
  sub x0, x1, #(Lend - Lstart)
  // CHECK: adds w0, w1, #8
  // CHECK: adds x0, x1, #8
  // CHECK: add w0, w1, #8
  // CHECK: add x0, x1, #8
  // CHECK: cmp w0, #8
  // CHECK: cmp x0, #8
  // CHECK: sub w0, w1, #8
  // CHECK: sub x0, x1, #8

  add w0, w1, #(Lend - Lstart), lsl #12
  cmp w0, #(Lend - Lstart), lsl #12
  // CHECK: add w0, w1, #8, lsl #12
  // CHECK: cmp w0, #8, lsl #12

  add w0, w1, #((Lend - Lstart) >> 2)
  cmp w0, #((Lend - Lstart) >> 2)
  // CHECK: add w0, w1, #2
  // CHECK: cmp w0, #2

  add w0, w1, #(Lend - Lstart + 12)
  cmp w0, #(Lend - Lstart + 12)
  // CHECK: add w0, w1, #20
  // CHECK: cmp w0, #20

  add w0, w1, #(Lforward - Lend)
  cmp w0, #(Lforward - Lend)
  // CHECK: add w0, w1, #320
  // CHECK: cmp w0, #320

  add w0, w1, #(Lstart - visible)
  cmp w0, #(Lstart - visible)
  // CHECK: add w0, w1, #8
  // CHECK: cmp w0, #8

// Add some filler so we don't have to modify #(Lforward - Lend) if we add more
// instructions above
Lfiller:
  .space 320 - (Lfiller - Lend)

Lforward:
  .space 4
  add w0, w1, #(. - Lforward)
  cmp w0, #(. - Lforward)
  // CHECK: add w0, w1, #4
  // CHECK: cmp w0, #8
