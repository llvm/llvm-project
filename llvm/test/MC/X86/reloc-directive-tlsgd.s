# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

## Test that splitting a fragment in two does not break a TLSGD sequence.

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.text {
# CHECK-NEXT:     0x0 R_X86_64_NONE .Lbase 0x0
# CHECK-NEXT:     0x3F58 R_X86_64_TLSGD tls 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:     0x3F60 R_X86_64_PLT32 __tls_get_addr 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.text
.Lbase:
        .rept 16212
        nop
        .endr
        data16
        leaq    tls@TLSGD(%rip), %rdi
        data16
        data16
        rex64
        callq   __tls_get_addr@PLT
        .reloc .Lbase, R_X86_64_NONE, .Lbase
