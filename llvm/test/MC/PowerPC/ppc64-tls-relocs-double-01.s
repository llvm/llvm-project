# RUN: llvm-mc -triple=powerpc64-unknown-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s

        .text
        addis 3, 13, myDouble@tprel@ha
        addi 3, 3, myDouble@tprel@l
        addis 3, 2, myDouble@got@tprel@ha
        ld 3, myDouble@got@tprel@l(3)
        lfdx 4, 3, myDouble@tls
        stfdx 4, 3, myDouble@tls
        .type myDouble,@object
        .section .tbss,"awT",@nobits
        .globl myDouble
        .align 2

myDouble:
	.quad	0
	.size	myDouble, 8

# Check for a pair of R_PPC64_TPREL16_HA / R_PPC64_TPREL16_LO relocs
# against the thread-local symbol 'myDouble'.
# CHECK:      Relocations [
# CHECK:        Section ({{[0-9]+}}) .rela.text {
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TPREL16_HA myDouble
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TPREL16_LO myDouble
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TPREL16_HA myDouble 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TPREL16_LO_DS myDouble 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS myDouble 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TLS myDouble 0x0
# CHECK-NEXT:   }
