// RUN: llvm-mc -triple aarch64-elf -filetype=obj %s -o - | llvm-objdump -d - | FileCheck %s

start:
  .space 8
end:
  // CHECK-LABEL: end:

  adds w0, w1, #(end - start)
  adds x0, x1, #(end - start)
  add w0, w1, #(end - start)
  add x0, x1, #(end - start)
  cmp w0, #(end - start)
  cmp x0, #(end - start)
  sub w0, w1, #(end - start)
  sub x0, x1, #(end - start)
  // CHECK: adds w0, w1, #8
  // CHECK: adds x0, x1, #8
  // CHECK: add w0, w1, #8
  // CHECK: add x0, x1, #8
  // CHECK: cmp w0, #8
  // CHECK: cmp x0, #8
  // CHECK: sub w0, w1, #8
  // CHECK: sub x0, x1, #8

  add w0, w1, #(end - start), lsl #12
  cmp w0, #(end - start), lsl #12
  // CHECK: add w0, w1, #8, lsl #12
  // CHECK: cmp w0, #8, lsl #12

  add w0, w1, #((end - start) >> 2)
  cmp w0, #((end - start) >> 2)
  // CHECK: add w0, w1, #2
  // CHECK: cmp w0, #2

  add w0, w1, #(end - start + 12)
  cmp w0, #(end - start + 12)
  // CHECK: add w0, w1, #20
  // CHECK: cmp w0, #20

  add w0, w1, #(forward - end)
  cmp w0, #(forward - end)
  // CHECK: add w0, w1, #320
  // CHECK: cmp w0, #320

// Add some filler so we don't have to modify #(forward - end) if we add more
// instructions above
.Lfiller:
  .space 320 - (.Lfiller - end)

forward:
  .space 8

.Lstart:
  .space 8
.Lend:
  add w0, w1, #(.Lend - .Lstart)
  cmp w0, #(.Lend - .Lstart)
  // CHECK: add w0, w1, #8
  // CHECK: cmp w0, #8

.Lprivate1:
  .space 8
notprivate:
  .space 8
.Lprivate2:
  add w0, w1, #(.Lprivate2 - .Lprivate1)
  cmp w0, #(.Lprivate2 - .Lprivate1)
  // CHECK: add w0, w1, #16
  // CHECK: cmp w0, #16

  .type foo, @function
foo:
  // CHECK-LABEL: foo:

  add w0, w1, #(foo - .Lprivate2)
  cmp w0, #(foo - .Lprivate2)
  // CHECK: add w0, w1, #8
  // CHECK: cmp w0, #8

  ret

  .type goo, @function
goo:
  // CHECK-LABEL: goo:

  add w0, w1, #(goo - foo)
  cmp w0, #(goo - foo)
  // CHECK: add w0, w1, #12
  // CHECK: cmp w0, #12

  add w0, w1, #(. - goo)
  cmp w0, #(. - goo)
  // CHECK: add w0, w1, #8
  // CHECK: cmp w0, #12

  ret
