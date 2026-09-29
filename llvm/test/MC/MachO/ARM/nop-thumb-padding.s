@ RUN: llvm-mc -triple armv6-apple-darwin %s -filetype=obj -o %t.obj
@ RUN: llvm-readobj -S --sd - < %t.obj > %t.dump
@ RUN: FileCheck %s < %t.dump

.thumb_func x
.code 16
x:
      adds r0, r1, r2
      .align 4
      adds r0, r1, r2

@ CHECK:  SectionData (
@ CHECK:    0000: 8818C046 C046C046 C046C046 C046C046  |...F.F.F.F.F.F.F|
@ CHECK:    0010: 8818                                 |..|
@ CHECK:  )
