# RUN: llvm-mc -triple=powerpc64-unknown-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -r | FileCheck %s

        .text
        addis 3, 13, t@tprel@ha
        addi 3, 3, t@tprel@l
        addis 3, 2, t@got@tprel@ha
        ld 3, t@got@tprel@l(3)
        lwzx 4, 3, t@tls
        lhzx 4, 3, t@tls
        lbzx 4, 3, t@tls
        ldx 4, 3, t@tls
        stbx 4, 3, t@tls
        sthx 4, 3, t@tls
        stwx 4, 3, t@tls
        stdx 4, 3, t@tls
        .type t,@object
        .section .tbss,"awT",@nobits
        .globl t
        .align 2
t:
        .long 0
        .size t, 4

# Check for a pair of R_PPC64_TPREL16_HA / R_PPC64_TPREL16_LO relocs
# against the thread-local symbol 't'.
# CHECK:      Relocations [
# CHECK:        Section ({{[0-9]+}}) .rela.text {
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TPREL16_HA t
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TPREL16_LO t
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TPREL16_HA t 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TPREL16_LO_DS t 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS t 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS t 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS t 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS t 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS t 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS t 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS t 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS t 0x0
# CHECK-NEXT:   }
