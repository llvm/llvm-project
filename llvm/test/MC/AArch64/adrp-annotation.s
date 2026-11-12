; RUN: llvm-mc -triple aarch64-apple-ios %s -filetype=obj -o %t.o
; RUN: llvm-objdump --macho -d %t.o | FileCheck %s

  .data_region
  .space 0x4124
  .end_data_region

  ; CHECK: 4124{{.*}}adrp x0, 5 ; 0x9000
  adrp x0, #0x5000
