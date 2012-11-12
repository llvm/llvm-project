@ RUN: llvm-mc -triple=thumbv7-apple-darwin -show-encoding %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

        ble Lfoo        @ wide encoding

        .space 258
Lfoo:
        nop

        ble Lbaz        @ narrow encoding
        .space 256
Lbaz:

@ CHECK: '_section_data', '40f38180
@ CHECK: 000000bf 7fdd
