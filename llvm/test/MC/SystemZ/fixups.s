
# RUN: llvm-mc -triple s390x-unknown-unknown --show-encoding %s | FileCheck %s

# RUN: llvm-mc -triple s390x-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r | FileCheck %s -check-prefix=CHECK-REL

# CHECK: larl %r14, target                      # encoding: [0xc0,0xe0,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target+2, kind: FK_390_PC32DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PC32DBL target 0x2
	.align 16
	larl %r14, target

# CHECK: larl %r14, target@GOT                  # encoding: [0xc0,0xe0,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@GOT+2, kind: FK_390_PC32DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_GOTENT target 0x2
	.align 16
	larl %r14, target@got

# CHECK: larl %r14, target@INDNTPOFF            # encoding: [0xc0,0xe0,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@INDNTPOFF+2, kind: FK_390_PC32DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_TLS_IEENT target 0x2
	.align 16
	larl %r14, target@indntpoff

# CHECK: brasl %r14, target                     # encoding: [0xc0,0xe5,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target+2, kind: FK_390_PC32DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PC32DBL target 0x2
	.align 16
	brasl %r14, target

# CHECK: brasl %r14, target@PLT                 # encoding: [0xc0,0xe5,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC32DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT32DBL target 0x2
	.align 16
	brasl %r14, target@plt

# CHECK: brasl %r14, target@PLT:tls_gdcall:sym  # encoding: [0xc0,0xe5,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC32DBL
# CHECK-NEXT:                                   # fixup B - offset: 0, value: sym@TLSGD, kind: FK_390_TLS_CALL
# CHECK-REL:                                    0x{{[0-9A-F]*0}} R_390_TLS_GDCALL sym 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT32DBL target 0x2
	.align 16
	brasl %r14, target@plt:tls_gdcall:sym

# CHECK: brasl %r14, target@PLT:tls_ldcall:sym  # encoding: [0xc0,0xe5,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC32DBL
# CHECK-NEXT:                                   # fixup B - offset: 0, value: sym@TLSLDM, kind: FK_390_TLS_CALL
# CHECK-REL:                                    0x{{[0-9A-F]*0}} R_390_TLS_LDCALL sym 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT32DBL target 0x2
	.align 16
	brasl %r14, target@plt:tls_ldcall:sym

# CHECK: bras %r14, target                      # encoding: [0xa7,0xe5,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target+2, kind: FK_390_PC16DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PC16DBL target 0x2
	.align 16
	bras %r14, target

# CHECK: bras %r14, target@PLT                  # encoding: [0xa7,0xe5,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC16DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT16DBL target 0x2
	.align 16
	bras %r14, target@plt

# CHECK: bras %r14, target@PLT:tls_gdcall:sym   # encoding: [0xa7,0xe5,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC16DBL
# CHECK-NEXT:                                   # fixup B - offset: 0, value: sym@TLSGD, kind: FK_390_TLS_CALL
# CHECK-REL:                                    0x{{[0-9A-F]*0}} R_390_TLS_GDCALL sym 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT16DBL target 0x2
	.align 16
	bras %r14, target@plt:tls_gdcall:sym

# CHECK: bras %r14, target@PLT:tls_ldcall:sym   # encoding: [0xa7,0xe5,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC16DBL
# CHECK-NEXT:                                   # fixup B - offset: 0, value: sym@TLSLDM, kind: FK_390_TLS_CALL
# CHECK-REL:                                    0x{{[0-9A-F]*0}} R_390_TLS_LDCALL sym 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT16DBL target 0x2
	.align 16
	bras %r14, target@plt:tls_ldcall:sym


# Data relocs
# llvm-mc does not show any "encoding" string for data, so we just check the relocs

# CHECK-REL: .rela.data
	.data

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LE64 target 0x0
	.align 16
	.quad target@ntpoff

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LDO64 target 0x0
	.align 16
	.quad target@dtpoff

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LDM64 target 0x0
	.align 16
	.quad target@tlsldm

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_GD64 target 0x0
	.align 16
	.quad target@tlsgd

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LE32 target 0x0
	.align 16
	.long target@ntpoff

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LDO32 target 0x0
	.align 16
	.long target@dtpoff

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LDM32 target 0x0
	.align 16
	.long target@tlsldm

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_GD32 target 0x0
	.align 16
	.long target@tlsgd

