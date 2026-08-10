# RUN: llvm-mc -triple i386-pc-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-readelf -S %t | FileCheck %s

# CHECK:      Name              Type            {{.*}} Flg Lk Inf Al
# CHECK:      .note             NOTE            {{.*}}      0   0  1
# CHECK-NEXT: .note2            NOTE            {{.*}}      0   0  1
# CHECK-NEXT: .notefoo          NOTE            {{.*}}      0   0  1
# CHECK-NEXT: .rodata.foo       PROGBITS        {{.*}}   A  0   0  1
# CHECK-NEXT: .rodatafoo        PROGBITS        {{.*}}      0   0  1
# CHECK-NEXT: .rodata1          PROGBITS        {{.*}}   A  0   0  1
# CHECK-NEXT: .tdata.foo        PROGBITS        {{.*}} WAT  0   0  1
# CHECK-NEXT: .tbss             NOBITS          {{.*}} WAT  0   0  1
# CHECK-NEXT: .tbss.foo         NOBITS          {{.*}} WAT  0   0  1
# CHECK-NEXT: .init_array       INIT_ARRAY      {{.*}}  WA  0   0  1
# CHECK-NEXT: .init_array.42    INIT_ARRAY      {{.*}}  WA  0   0  1
# CHECK-NEXT: .init_array2      PROGBITS        {{.*}}      0   0  1
# CHECK-NEXT: .init_arrayfoo    PROGBITS        {{.*}}      0   0  1
# CHECK-NEXT: .fini_array       FINI_ARRAY      {{.*}}  WA  0   0  1
# CHECK-NEXT: .fini_array2      PROGBITS        {{.*}}      0   0  1
# CHECK-NEXT: .fini_arrayfoo    PROGBITS        {{.*}}      0   0  1
# CHECK-NEXT: .preinit_array    PREINIT_ARRAY   {{.*}}  WA  0   0  1
# CHECK-NEXT: .preinit_array2   PROGBITS        {{.*}}      0   0  1
# CHECK-NEXT: .preinit_array.x  PREINIT_ARRAY   {{.*}}  WA  0   0  1
# CHECK-NEXT: .data.foo         PROGBITS        {{.*}}  WA  0   0  1
# CHECK-NEXT: .data1            PROGBITS        {{.*}}  WA  0   0  1
# CHECK-NEXT: .data2            PROGBITS        {{.*}}      0   0  1
# CHECK-NEXT: .bss              NOBITS          {{.*}}  WA  0   0  1
# CHECK-NEXT: .bss.foo          NOBITS          {{.*}}  WA  0   0  1
# CHECK-NEXT: .nobits           PROGBITS        {{.*}}      0   0  1
# CHECK-NEXT: .nobits2          PROGBITS        {{.*}}      0   0  1
# CHECK-NEXT: .nobitsfoo        PROGBITS        {{.*}}      0   0  1


.section .note
.section .note2
.section .notefoo

.section .rodata.foo
.section .rodatafoo
.section .rodata1

.section .tdata.foo
.section .tbss
.section .tbss.foo

.section .init_array
.section .init_array.42
.section .init_array2
.section .init_arrayfoo
.section .fini_array
.section .fini_array2
.section .fini_arrayfoo
.section .preinit_array
.section .preinit_array2
.section .preinit_array.x

.section .data.foo
.section .data1
.section .data2
.section .bss
.section .bss.foo

.section .nobits
.section .nobits2
.section .nobitsfoo
.byte 1
