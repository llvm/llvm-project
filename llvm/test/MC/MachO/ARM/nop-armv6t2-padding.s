@ RUN: llvm-mc -triple armv6t2-apple-darwin %s -filetype=obj -o %t.obj
@ RUN: llvm-readobj -S --sd - < %t.obj > %t.dump
@ RUN: FileCheck %s < %t.dump

x:
      add r0, r1, r2
      .align 4
      add r0, r1, r2

@ CHECK:  SectionData (
@ CHECK:    0000: 020081E0 00F020E3 00F020E3 00F020E3  |...... ... ... .|
@ CHECK:    0010: 020081E0                             |....|
@ CHECK:  )
