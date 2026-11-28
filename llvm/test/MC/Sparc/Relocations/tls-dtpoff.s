# RUN: llvm-mc -triple=sparc %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple=sparcv9 %s --defsym V9=1 | FileCheck %s --check-prefixes=ASM,ASM64
# RUN: llvm-mc -triple=sparc -filetype=obj %s | llvm-readelf -rs - | FileCheck %s --check-prefixes=CHECK
# RUN: llvm-mc -triple=sparcv9 -filetype=obj %s --defsym V9=1 | llvm-readelf -rs - | FileCheck %s --check-prefixes=CHECK,CHECK64

# RUN: not llvm-mc -triple=sparcv9 -filetype=obj %s --defsym ERR=1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

## %r_tls_dtpoff32/%r_tls_dtpoff64 are used by GCC for the DWARF location of
## TLS variables. Like GNU as, they mark the referenced symbol STT_TLS.

# ASM:        .word %r_tls_dtpoff32(tv)
# ASM-NEXT:   .word %r_tls_dtpoff32(lv+4)
# ASM-NEXT:   .word %r_tls_dtpoff32(ext32)
# ASM64:      .xword %r_tls_dtpoff64(tv)
# ASM64-NEXT: .xword %r_tls_dtpoff64(lv+4)
# ASM64-NEXT: .xword %r_tls_dtpoff64(ext64)

# CHECK:        Relocation section '.rela.debug_info'
# CHECK:        R_SPARC_TLS_DTPOFF32 {{0+}} tv + 0
# CHECK-NEXT:   R_SPARC_TLS_DTPOFF32 {{0+}}4 lv + 4
# CHECK-NEXT:   R_SPARC_TLS_DTPOFF32 {{0+}} ext32 + 0
# CHECK64-NEXT: R_SPARC_TLS_DTPOFF64 {{0+}} tv + 0
# CHECK64-NEXT: R_SPARC_TLS_DTPOFF64 {{0+}}4 lv + 4
# CHECK64-NEXT: R_SPARC_TLS_DTPOFF64 {{0+}} ext64 + 0

# CHECK:     TLS LOCAL DEFAULT {{.*}} lv
# CHECK:     TLS GLOBAL DEFAULT {{.*}} tv
# CHECK:     TLS GLOBAL DEFAULT UND ext32
# CHECK64:   TLS GLOBAL DEFAULT UND ext64

.section .tbss,"awT",@nobits
.globl tv
tv:
  .zero 4
lv:
  .zero 8

.section .debug_info,"",@progbits
.word %r_tls_dtpoff32(tv)
.word %r_tls_dtpoff32(lv + 4)
.word %r_tls_dtpoff32(ext32)

.ifdef V9
.xword %r_tls_dtpoff64(tv)
.xword %r_tls_dtpoff64(lv + 4)
.xword %r_tls_dtpoff64(ext64)
.endif

.ifdef ERR
# ERR: [[#@LINE+1]]:9: error: %r_tls_dtpoff32 can only be used in a .word directive
.xword %r_tls_dtpoff32(tv)
# ERR: [[#@LINE+1]]:8: error: %r_tls_dtpoff64 can only be used in a .xword directive
.word %r_tls_dtpoff64(tv)
.endif
