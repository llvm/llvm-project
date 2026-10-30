# RUN: llvm-mc -triple=riscv32 -filetype=obj %s | llvm-readelf -Ss - | FileCheck %s --check-prefix=CHECK,CHECK-RV32
# RUN: llvm-mc -triple=riscv64 -filetype=obj %s | llvm-readelf -Ss - | FileCheck %s --check-prefix=CHECK,CHECK-RV64

        .text
        nop

# .wibble should *not* inherit .text's mapping symbol. It's a completely
# different section.
        .section .wibble
        nop

# A section should be able to start with a $d.
        .section .starts_data
        .word 42

# Changing back to .text should not emit a redundant $x - the active ISA
# has not changed since the last mapping symbol emitted in .text.
        .text
        nop

# Pushing a data section and popping back to .text should also preserve .text's
# mapping symbol state and not emit a redundant $x.
        .pushsection .starts_data
        .word 42
        .popsection
        nop

# With all those constraints, we want:
#   + .text to have $x<ISA> at 0 and no others
#   + .wibble to have $x<ISA> at 0 (each code section records the active ISA
#   + .starts_data to have $d at 0

## Capture section indices.
# CHECK: [[#TEXT:]]] .text
# CHECK: [[#WIBBLE:]]] .wibble
# CHECK: [[#STARTS_DATA:]]] .starts_data

# CHECK:           Symbol table '.symtab' contains 4 entries:
# CHECK-NEXT:         Num:    Value  Size Type    Bind   Vis     Ndx Name
# CHECK-NEXT:           0: {{0+}}       0 NOTYPE  LOCAL  DEFAULT   UND {{$}}
# CHECK-RV32-NEXT:      1: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]]        $xrv32i2p1{{$}}
# CHECK-RV64-NEXT:      1: {{0+}}       0 NOTYPE  LOCAL  DEFAULT [[#TEXT]]        $xrv64i2p1{{$}}
# CHECK-RV32-NEXT:      2: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#WIBBLE]]      $xrv32i2p1{{$}}
# CHECK-RV64-NEXT:      2: {{0+}}       0 NOTYPE  LOCAL  DEFAULT [[#WIBBLE]]      $xrv64i2p1{{$}}
# CHECK-NEXT:           3: {{0+}}       0 NOTYPE  LOCAL  DEFAULT [[#STARTS_DATA]] $d{{$}}
# CHECK-NOT:       {{.}}
