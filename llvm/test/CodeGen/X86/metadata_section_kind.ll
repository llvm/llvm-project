; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=CHECK-ELF
; RUN: llc < %s -mtriple=x86_64-win32-gnu | FileCheck %s --check-prefix=CHECK-COFF

@a = private constant [1 x i8] c"\00", section ".test", align 8, !metadata_section_kind !{}

;;              section: name, flags, type
; CHECK-ELF:   .section  .test,"",@progbits
; CHECK-COFF:  .section  .test,"yD"
