# RUN: llvm-mc -triple i686-windows-gnu -g %s -filetype obj -o - \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix CHECK-COFF %s
# RUN: llvm-mc -triple i686-windows-itanium -g %s -filetype obj -o - \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix CHECK-COFF %s
# RUN: llvm-mc -triple i686-linux-gnu -g %s -filetype obj -o - \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix CHECK-ELF %s

_a:
	movl $65, %eax
	ret

# CHECK-COFF: Relocations [
# CHECK-COFF:   Section {{.*}} .debug_info {
# CHECK-COFF:     0x6 IMAGE_REL_I386_SECREL .debug_abbrev
# CHECK-COFF:     0xC IMAGE_REL_I386_SECREL .debug_line
# CHECK-COFF:   }
# CHECK-COFF:   Section {{.*}} .debug_aranges {
# CHECK-COFF:     0x6 IMAGE_REL_I386_SECREL .debug_info
# CHECK-COFF:   }
# CHECK-COFF: ]

# CHECK-ELF: Relocations [
# CHECK-ELF:   Section {{.*}} .rel.debug_info {
# CHECK-ELF:     0x6 R_386_32 .debug_abbrev
# CHECK-ELF:     0xC R_386_32 .debug_line
# CHECK-ELF:   }
# CHECK-ELF:   Section {{.*}} .rel.debug_aranges {
# CHECK-ELF:     0x6 R_386_32 .debug_info
# CHECK-ELF:   }
# CHECK-ELF: ]
