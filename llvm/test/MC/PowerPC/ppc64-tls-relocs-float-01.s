# RUN: llvm-mc -triple=powerpc64-unknown-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s

        .text
        addis 3, 13, myFloat@tprel@ha
        addi 3, 3, myFloat@tprel@l
        addis 3, 2, myFloat@got@tprel@ha
        ld 3, myFloat@got@tprel@l(3)
        lfsx 4, 3, myFloat@tls
        stfsx 4, 3, myFloat@tls
        .type myFloat,@object
        .section .tbss,"awT",@nobits
        .globl myFloat
        .align 2

myFloat:
	.long	0
	.size	myFloat, 4

# Check for a pair of R_PPC64_TPREL16_HA / R_PPC64_TPREL16_LO relocs
# against the thread-local symbol 'myFloat'.
# CHECK:      Relocations [
# CHECK:        Section ({{[0-9]+}}) .rela.text {
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TPREL16_HA myFloat
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TPREL16_LO myFloat
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TPREL16_HA myFloat 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TPREL16_LO_DS myFloat 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS myFloat 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS myFloat 0x0
# CHECK-NEXT:   }
