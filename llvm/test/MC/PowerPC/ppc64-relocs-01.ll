;; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -O3  \
;; RUN:  -filetype=obj %s -o - | \
;; RUN: elf-dump --dump-section-data | FileCheck %s

;; FIXME: this file need to be in .s form, change when asm parse is done.

@number64 = global i64 10, align 8

define i64 @access_int64(i64 %a) nounwind readonly {
entry:
  %0 = load i64* @number64, align 8
  %cmp = icmp eq i64 %0, %a
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

declare double @sin(double) nounwind

define double @test_branch24 (double %x) nounwind readonly {
entry:
  %add = call double @sin(double %x) nounwind
  ret double %add
}

;; The relocations in .rela.text are the 'number64' load using a
;; R_PPC64_TOC16_DS against the .toc and the 'sin' external function
;; address using a R_PPC64_REL24
;; CHECK:       '.rela.text'
;; CHECK:       Relocation 0
;; CHECK-NEXT:  'r_offset',
;; CHECK-NEXT:  'r_sym', 0x00000006
;; CHECK-NEXT:  'r_type', 0x0000003f
;; CHECK:       Relocation 1
;; CHECK-NEXT:  'r_offset',
;; CHECK-NEXT:  'r_sym', 0x0000000a
;; CHECK-NEXT:  'r_type', 0x0000000a

;; The .opd entry for the 'access_int64' function creates 2 relocations:
;; 1. A R_PPC64_ADDR64 against the .text segment plus addend (the function
;    address itself);
;; 2. And a R_PPC64_TOC against no symbol (the linker will replace for the
;;    module's TOC base).
;; CHECK:       '.rela.opd'
;; CHECK:       Relocation 0
;; CHECK-NEXT:  'r_offset',
;; CHECK-NEXT:  'r_sym', 0x00000002
;; CHECK-NEXT:  'r_type', 0x00000026
;; CHECK:       Relocation 1
;; CHECK-NEXT:  'r_offset',
;; CHECK-NEXT:  'r_sym', 0x00000000
;; CHECK-NEXT:  'r_type', 0x00000033

;; Finally the TOC creates the relocation for the 'number64'.
;; CHECK:       '.rela.toc'
;; CHECK:       Relocation 0
;; CHECK-NEXT:  'r_offset',
;; CHECK-NEXT:  'r_sym', 0x00000008
;; CHECK-NEXT:  'r_type', 0x00000026

;; Check if the relocation references are for correct symbols.
;; CHECK:       Symbol 7
;; CHECK-NEXT:  'access_int64'
;; CHECK:       Symbol 8
;; CHECK-NEXT:  'number64'
;; CHECK:       Symbol 10
;; CHECK-NEXT:  'sin'
