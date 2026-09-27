; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=CHECK-ELF
; RUN: llc < %s -mtriple=x86_64-win32-gnu | FileCheck %s --check-prefix=CHECK-COFF

@a = private constant [1 x i8] c"\00", section ".test", align 8, !metadata_section_kind !{}
@b = private constant [1 x i8] c"\00", section ".test2", align 8

;;              section: name, flags, type
; CHECK-ELF:   .section  .test,"",@progbits
; CHECK-COFF:  .section  .test,"yD"
;; Check no metadata results in different output to prevent rotten
;; green test.
; CHECK-ELF:   .section  .test2,"a",@progbits
; CHECK-COFF:  .section  .test2,"dr"
