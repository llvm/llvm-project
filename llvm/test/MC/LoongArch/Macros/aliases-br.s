# RUN: llvm-mc --triple=loongarch64 %s | FileCheck %s

bgt   $a1, $a0, 16
# CHECK:      blt     $a0, $a1, 16
bgtu  $a1, $a0, 16
# CHECK-NEXT: bltu    $a0, $a1, 16
ble   $a1, $a0, 16
# CHECK-NEXT: bge     $a0, $a1, 16
bleu  $a1, $a0, 16
# CHECK-NEXT: bgeu    $a0, $a1, 16
bltz   $a0, 16
# CHECK-NEXT: bltz    $a0, 16
bgtz   $a0, 16
# CHECK-NEXT: bgtz    $a0, 16
blez   $a0, 16
# CHECK-NEXT: blez    $a0, 16
bgez   $a0, 16
# CHECK-NEXT: bgez    $a0, 16
