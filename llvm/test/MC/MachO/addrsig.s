# REQUIRES: aarch64-registered-target
# RUN: llvm-mc -filetype=obj -triple arm64-apple-darwin %s -o %t
# RUN: llvm-readobj -r -s %t | FileCheck %s

# CHECK:      Section __llvm_addrsig {
# CHECK-NEXT:   0x0 0 3 1 ARM64_RELOC_UNSIGNED 0 .Llocal
# CHECK-NEXT:   0x0 0 3 1 ARM64_RELOC_UNSIGNED 0 local
# CHECK-NEXT:   0x0 0 3 1 ARM64_RELOC_UNSIGNED 0 g3
# CHECK-NEXT:   0x0 0 3 1 ARM64_RELOC_UNSIGNED 0 g1
# CHECK-NEXT: }

# CHECK:      Symbols [
# CHECK:        Symbol {
# CHECK-NEXT:     Name: ltmp0
# CHECK:        Symbol {
# CHECK-NEXT:     Name: local
# CHECK:        Symbol {
# CHECK-NEXT:     Name: .Llocal
# CHECK:        Symbol {
# CHECK-NEXT:     Name: ltmp1
# CHECK:        Symbol {
# CHECK-NEXT:     Name: g1
# CHECK:        Symbol {
# CHECK-NEXT:     Name: g2
# CHECK:        Symbol {
# CHECK-NEXT:     Name: g3
# CHECK-NOT:    Symbol {
# CHECK:      ]

.globl g1

.addrsig
.addrsig_sym g1
.globl g2
.addrsig_sym g3
.addrsig_sym local
.addrsig_sym .Llocal

local:
  nop
.globl g3

.data
.Llocal:

.subsections_via_symbols
