# RUN: llvm-mc -triple=powerpc %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# RUN: not llvm-mc -triple=powerpc --defsym ERR=1 %s 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

# ASM: bl __tls_get_addr(a@tlsgd)
# ASM: bl __tls_get_addr(b@tlsld)
# ASM: bl __tls_get_addr(c@tlsgd)@PLT
# ASM: bl __tls_get_addr(d@tlsld)@PLT+32768
# ASM: bl __tls_get_addr(e@tlsld)@PLT+32768
bl __tls_get_addr(a@tlsgd)
bl __tls_get_addr(b@tlsld)
bl __tls_get_addr(c@tlsgd)@plt
bl __tls_get_addr(d@tlsld)@PLT+32768
bl __tls_get_addr+32768(e@tlsld)@plt  # gcc -fPIC

## These are not present in the wild, but just to test we can parse them.
# ASM: bl __tls_get_addr(f@tlsld)@PLT+1+(-2)
bl __tls_get_addr+1(f@tlsld)@PLT+-2
# ASM: bl __tls_get_addr(g@tlsld)@PLT+1+(y-x)
x:
bl __tls_get_addr+1(g@tlsld)@PLT+(y-x)
y:

# CHECK:      .rela.text {
# CHECK-NEXT:   0x0 R_PPC_TLSGD a 0x0
# CHECK-NEXT:   0x0 R_PPC_REL24 __tls_get_addr 0x0
# CHECK-NEXT:   0x4 R_PPC_TLSLD b 0x0
# CHECK-NEXT:   0x4 R_PPC_REL24 __tls_get_addr 0x0
# CHECK-NEXT:   0x8 R_PPC_TLSGD c 0x0
# CHECK-NEXT:   0x8 R_PPC_PLTREL24 __tls_get_addr 0x0
# CHECK-NEXT:   0xC R_PPC_TLSLD d 0x0
# CHECK-NEXT:   0xC R_PPC_PLTREL24 __tls_get_addr 0x8000
# CHECK-NEXT:   0x10 R_PPC_TLSLD e 0x0
# CHECK-NEXT:   0x10 R_PPC_PLTREL24 __tls_get_addr 0x8000
# CHECK-NEXT:   0x14 R_PPC_TLSLD f 0x0
# CHECK-NEXT:   0x14 R_PPC_PLTREL24 __tls_get_addr 0xFFFFFFFF
# CHECK-NEXT:   0x18 R_PPC_TLSLD g 0x0
# CHECK-NEXT:   0x18 R_PPC_PLTREL24 __tls_get_addr 0x5
# CHECK-NEXT: }

.ifdef ERR
# ERR: :[[#@LINE+1]]:27: error: unexpected token
bl __tls_get_addr(d@tlsld)plt
# ERR: :[[#@LINE+1]]:28: error: expected 'plt'
bl __tls_get_addr(d@tlsld)@invalid
# ERR: :[[#@LINE+1]]:31: error: unexpected token
bl __tls_get_addr(d@tlsld)@plt-32768

# ERR: :[[#@LINE+1]]:21: error: invalid memory operand
bl __tls_get_addr-1(f@tlsld)@plt
.endif
