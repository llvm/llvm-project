@ RUN: llvm-mc -triple armv6-apple-darwin %s -filetype=obj -o %t.obj
@ RUN: macho-dump --dump-section-data < %t.obj > %t.dump
@ RUN: FileCheck %s < %t.dump

.thumb_func x
.code 16
x:
      adds r0, r1, r2
      .align 4
      adds r0, r1, r2

@ CHECK: ('_section_data', '8818c046 c046c046 c046c046 c046c046 8818')
