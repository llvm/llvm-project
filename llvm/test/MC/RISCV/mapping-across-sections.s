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

# With all those constraints, we want:
#   + .text to have $x<ISA> at 0 and no others
#   + .wibble to have $x<ISA> at 0 (each code section records the active ISA
#   + .starts_data to have $d at 0

## Capture section indices.
# CHECK: [[#TEXT:]]] .text
# CHECK: [[#WIBBLE:]]] .wibble
# CHECK: [[#STARTS_DATA:]]] .starts_data

# CHECK:    Value  Size Type    Bind   Vis     Ndx              Name
# CHECK-RV32: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]]        $xrv32i2p1{{$}}
# CHECK-RV64: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]]        $xrv64i2p1{{$}}
# CHECK-RV32: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#WIBBLE]]      $xrv32i2p1{{$}}
# CHECK-RV64: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#WIBBLE]]      $xrv64i2p1{{$}}
# CHECK: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#STARTS_DATA]] $d{{$}}
