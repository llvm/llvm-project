// RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o %t.o
f:
 .cfi_startproc
 .cfi_endproc

EH_frame0:
