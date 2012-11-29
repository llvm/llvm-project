;; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -filetype=obj %s -o - | \
;; RUN: elf-dump --dump-section-data | FileCheck %s

;; FIXME: this file should be in .s form, change when asm parser is available.

define void @f() {
entry:
  ret void
}

;; CHECK:      ('sh_name', 0x{{.*}}) # '.eh_frame'
;; CHECK-NEXT: ('sh_type', 0x00000001)
;; CHECK-NEXT: ('sh_flags', 0x0000000000000002)
;; CHECK-NEXT: ('sh_addr', 0x{{.*}})
;; CHECK-NEXT: ('sh_offset', 0x{{.*}})
;; CHECK-NEXT: ('sh_size', 0x0000000000000030)
;; CHECK-NEXT: ('sh_link', 0x00000000)
;; CHECK-NEXT: ('sh_info', 0x00000000)
;; CHECK-NEXT: ('sh_addralign', 0x0000000000000008)
;; CHECK-NEXT: ('sh_entsize', 0x0000000000000000)
;; CHECK-NEXT: ('_section_data', '00000010 00000000 017a5200 01784101 000c0100 00000018 00000018 00000000 00000000 00000000 00000010 00000000')

;; CHECK:      ('sh_name', 0x{{.*}}) # '.rela.eh_frame'
;; CHECK-NEXT: ('sh_type', 0x00000004)
;; CHECK-NEXT: ('sh_flags', 0x0000000000000000)
;; CHECK-NEXT: ('sh_addr', 0x{{.*}})
;; CHECK-NEXT: ('sh_offset', 0x{{.*}})
;; CHECK-NEXT: ('sh_size', 0x0000000000000018)
;; CHECK-NEXT: ('sh_link', 0x{{.*}})
;; CHECK-NEXT: ('sh_info', 0x{{.*}})
;; CHECK-NEXT: ('sh_addralign', 0x0000000000000008)
;; CHECK-NEXT: ('sh_entsize', 0x0000000000000018)
;; CHECK-NEXT: ('_relocations', [
;; CHECK-NEXT:  # Relocation 0
;; CHECK-NEXT:  (('r_offset', 0x000000000000001c)
;; CHECK-NEXT:   ('r_sym', 0x{{.*}})
;; CHECK-NEXT:   ('r_type', 0x00000026)
;; CHECK-NEXT:   ('r_addend', 0x0000000000000000)
;; CHECK-NEXT:  ),
;; CHECK-NEXT: ])

