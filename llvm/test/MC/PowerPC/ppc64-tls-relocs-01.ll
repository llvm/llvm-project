;; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -filetype=obj %s -o - | \
;; RUN: elf-dump --dump-section-data | FileCheck %s

;; FIXME: this file should be in .s form, change when asm parser is available.

@t = thread_local global i32 0, align 4

define i32* @f() nounwind {
entry:
  ret i32* @t
}

;; Check for a pair of R_PPC64_TPREL16_HA / R_PPC64_TPREL16_LO relocs
;; against the thread-local symbol 't'.
;; CHECK:       '.rela.text'
;; CHECK:       Relocation 0
;; CHECK-NEXT:  'r_offset',
;; CHECK-NEXT:  'r_sym', 0x00000008
;; CHECK-NEXT:  'r_type', 0x00000048
;; CHECK:       Relocation 1
;; CHECK-NEXT:  'r_offset',
;; CHECK-NEXT:  'r_sym', 0x00000008
;; CHECK-NEXT:  'r_type', 0x00000046

;; Check that we got the correct symbol.
;; CHECK:       Symbol 8
;; CHECK-NEXT:  't'

