# RUN: llvm-mc -triple=powerpc64le-unknown-unknown -filetype=obj %s 2>&1 | \
# RUN: FileCheck %s -check-prefix=MC
# RUN: llvm-mc -triple=powerpc64le-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s -check-prefix=READOBJ

# This test checks that on Power PC we can correctly convert @got@tlsld@pcrel
# x@tlsld, __tls_get_addr@notoc and x@DTPREL into R_PPC64_GOT_TLSLD_PCREL34,
# R_PPC64_TLSLD, R_PPC64_REL24_NOTOC and R_PPC64_DTPREL34 for local dynamic
# relocations with address/value loaded

# MC-NOT:  error: invalid variant

# READOBJ:       0x0 R_PPC64_GOT_TLSLD_PCREL34 x 0x0
# READOBJ-NEXT:  0x8 R_PPC64_TLSLD x 0x0
# READOBJ-NEXT:  0x8 R_PPC64_REL24_NOTOC __tls_get_addr 0x0
# READOBJ-NEXT:  0xC R_PPC64_DTPREL34 x 0x0
# READOBJ-NEXT:  0x18 R_PPC64_GOT_TLSLD_PCREL34 x 0x0
# READOBJ-NEXT:  0x20 R_PPC64_TLSLD x 0x0
# READOBJ-NEXT:  0x20 R_PPC64_REL24_NOTOC __tls_get_addr 0x0
# READOBJ-NEXT:  0x24 R_PPC64_DTPREL34 x 0x0

LocalDynamicAddrLoad:
  paddi 3, 0, x@got@tlsld@pcrel, 1
  bl __tls_get_addr@notoc(x@tlsld)
  paddi 3, 3, x@DTPREL, 0
  blr

LocalDynamicValueLoad:
  paddi 3, 0, x@got@tlsld@pcrel, 1
  bl __tls_get_addr@notoc(x@tlsld)
  paddi 3, 3, x@DTPREL, 0
  lwz 3, 0(3)
  blr
