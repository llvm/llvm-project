; RUN: llc < %s -filetype=obj -mtriple=thumbv6-apple-darwin -o - | macho-dump --dump-section-data | FileCheck -check-prefix=CHECK-T1 %s
; RUN: llc < %s -filetype=obj -mtriple=thumbv7-apple-darwin -o - | macho-dump --dump-section-data | FileCheck -check-prefix=CHECK-T2 %s
; RUN: llc < %s -filetype=obj -mtriple=armv6-apple-darwin -o - | macho-dump --dump-section-data | FileCheck -check-prefix=CHECK-ARM %s
; RUN: llc < %s -filetype=obj -mtriple=armv7-apple-darwin -o - | macho-dump --dump-section-data | FileCheck -check-prefix=CHECK-ARMV7 %s

; Empty functions need a NOP in them for MachO to prevent DWARF FDEs from
; getting all mucked up. See lib/CodeGen/AsmPrinter/AsmPrinter.cpp for
; details.
define internal fastcc void @empty_function() {
  unreachable
}
; CHECK-T1:    ('_section_data', 'c046')
; CHECK-T2:    ('_section_data', '00bf')
; CHECK-ARM:   ('_section_data', '0000a0e1')
; CHECK-ARMV7: ('_section_data', '00f020e3')
