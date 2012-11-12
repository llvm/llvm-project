@ RUN: llvm-mc -triple armv6t2-apple-darwin %s -filetype=obj -o %t.obj
@ RUN: macho-dump --dump-section-data < %t.obj > %t.dump
@ RUN: FileCheck %s < %t.dump

x:
      add r0, r1, r2
      .align 4
      add r0, r1, r2

@ CHECK: ('_section_data', '020081e0 00f020e3 00f020e3 00f020e3 020081e0')
