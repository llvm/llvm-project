# RUN: llvm-mc -triple i386-pc-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-readobj -S < %t | FileCheck %s
.section .nobits
.byte 1
.section .nobits2
.byte 1
.section .nobitsfoo
.byte 1
.section .init_array
.byte 1
.section .init_array.42
.byte 1
.section .init_array2
.byte 1
.section .init_arrayfoo
.byte 1
.section .fini_array
.byte 1
.section .fini_array2
.byte 1
.section .fini_arrayfoo
.byte 1
.section .preinit_array
.byte 1
.section .preinit_array2
.byte 1
.section .preinit_arrayfoo
.byte 1
.section .note
.byte 1
.section .note2
.byte 1
.section .notefoo
.byte 1
.section .bss
.space 1
.section .bss.foo
.space 1
.section .tbss
.space 1
.section .tbss.foo
.space 1
# CHECK:        Name: .nobits
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK:        Name: .nobits2
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK:        Name: .nobitsfoo
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK:        Name: .init_array
# CHECK-NEXT:   Type:  SHT_INIT_ARRAY
# CHECK:        Name: .init_array.42
# CHECK-NEXT:   Type:  SHT_INIT_ARRAY
# CHECK:        Name: .init_array2
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK:        Name: .init_arrayfoo
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK:        Name: .fini_array
# CHECK-NEXT:   Type: SHT_FINI_ARRAY
# CHECK:        Name: .fini_array2
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK:        Name: .fini_arrayfoo
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK:        Name: .preinit_array
# CHECK-NEXT:   Type: SHT_PREINIT_ARRAY
# CHECK:        Name: .preinit_array2
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK:        Name: .preinit_arrayfoo
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK:        Name: .note
# CHECK-NEXT:   Type: SHT_NOTE
# CHECK:        Name: .note2
# CHECK-NEXT:   Type: SHT_NOTE
# CHECK:        Name: .notefoo
# CHECK-NEXT:   Type: SHT_NOTE
# CHECK:        Name: .bss
# CHECK-NEXT:   Type: SHT_NOBITS
# CHECK:        Name: .bss.foo
# CHECK-NEXT:   Type: SHT_NOBITS
# CHECK:        Name: .tbss
# CHECK-NEXT:   Type: SHT_NOBITS
# CHECK:        Name: .tbss.foo
# CHECK-NEXT:   Type: SHT_NOBITS
