@ RUN: llvm-mc -triple armv7-apple-darwin %s -filetype=obj -o %t.obj
@ RUN: llvm-readobj -S --sd < %t.obj > %t.dump
@ RUN: FileCheck %s < %t.dump

.thumb_func x
.code 16
x:
      adds r0, r1, r2
      .align 4
      adds r0, r1, r2

@ CHECK:  SectionData (
@ CHECK:    0000: 881800BF 00BF00BF 00BF00BF 00BF00BF  |................|
@ CHECK:    0010: 8818                                 |..|
@ CHECK:  )
