# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s -o %t.o
# RUN: llvm-objdump -dr --no-show-raw-insn %t.o | FileCheck %s

  .text
  .type   test,@function
test:
  nop
.L1:
  .fill 0x100000, 4, 0x0

## R_LARCH_B16

# CHECK:         bne     $t0, $t1, 8
# CHECK-NEXT:    b       -4194308
  beq $t0, $t1, .L1

# CHECK:         beq     $t0, $t1, 8
# CHECK-NEXT:    b       -4194316
  bne $t0, $t1, .L1

# CHECK:         bge     $t0, $t1, 8
# CHECK-NEXT:    b       -4194324
  blt $t0, $t1, .L1

# CHECK:         bge     $t1, $t0, 8
# CHECK-NEXT:    b       -4194332
  bgt $t0, $t1, .L1

# CHECK:         bge     $t0, $zero, 8
# CHECK-NEXT:    b       -4194340
  bltz $t0, .L1

# CHECK:         bge     $zero, $t0, 8
# CHECK-NEXT:    b       -4194348
  bgtz $t0, .L1

# CHECK:         blt     $t1, $t0, 8
# CHECK-NEXT:    b       -4194356
  ble $t0, $t1, .L1

# CHECK:         blt     $t0, $t1, 8
# CHECK-NEXT:    b       -4194364
  bge $t0, $t1, .L1

# CHECK:         blt     $zero, $t0, 8
# CHECK-NEXT:    b       -4194372
  blez $t0, .L1

# CHECK:         blt     $t0, $zero, 8
# CHECK-NEXT:    b       -4194380
  bgez $t0, .L1

# CHECK:         bgeu    $t0, $t1, 8
# CHECK-NEXT:    b       -4194388
  bltu $t0, $t1, .L1

# CHECK:         bgeu    $t1, $t0, 8
# CHECK-NEXT:    b       -4194396
  bgtu $t0, $t1, .L1

# CHECK:         bltu    $t1, $t0, 8
# CHECK-NEXT:    b       -4194404
  bleu $t0, $t1, .L1

# CHECK:         bltu    $t0, $t1, 8
# CHECK-NEXT:    b       -4194412
  bgeu $t0, $t1, .L1

## R_LARCH_B21

# CHECK:         bnez    $t0, 8
# CHECK-NEXT:    b       -4194420
  beqz $t0, .L1

# CHECK:         beqz    $t0, 8
# CHECK-NEXT:    b       -4194428
  bnez $t0, .L1

# CHECK:         bcnez   $fcc0, 8
# CHECK-NEXT:    b       -4194436
  bceqz $fcc0, .L1

# CHECK:         bceqz   $fcc0, 8
# CHECK-NEXT:    b       -4194444
  bcnez $fcc0, .L1

## Not relax if symbol is unresolved
# CHECK:         bnez    $t0, 0
# CHECK-NEXT:    R_LARCH_B21  foo
# CHECK-NEXT:    bnez    $t0, 0
# CHECK-NEXT:    R_LARCH_B21  .text2
# CHECK-NEXT:    ret
  bnez $t0, foo
  bnez $t0, test2
  ret
.Lfunc_end0:
  .size test, .Lfunc_end0-test

  .section .text2, "ax"
  .type   test2,@function
test2:
  ret
.Lfunc_end1:
  .size test2, .Lfunc_end1-test2
