! Testing Sparc TLS relocations emission
! (for now a couple local ones).
!
! RUN: llvm-mc %s -arch=sparc -show-encoding | FileCheck %s --check-prefix=ASM
! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s --check-prefix=ASM
! RUN: llvm-mc %s -arch=sparc -filetype=obj | llvm-readobj -r | FileCheck %s --check-prefix=REL
! RUN: llvm-mc %s -arch=sparcv9 -filetype=obj | llvm-readobj -r | FileCheck %s --check-prefix=REL
! RUN: llvm-mc %s -arch=sparc -filetype=obj | llvm-objdump -r -d - | FileCheck %s --check-prefix=OBJDUMP
! RUN: llvm-mc %s -arch=sparcv9 -filetype=obj | llvm-objdump -r -d - | FileCheck %s --check-prefix=OBJDUMP

! REL: Arch: sparc
! REL: Relocations [
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LE_HIX22 Head 0x0
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LE_LOX10 Head 0x0
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LDO_HIX22 Head 0x0
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LDM_HI22  Head 0x0
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LDM_LO10  Head 0x0
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LDO_LOX10 Head 0x0
! REL: ]


! OBJDUMP: foo:
foo:
! Here we use two different sequences to get the address of a static TLS variable 'Head'
! (note - there is no intent to have valid assembler function here,
!  we just check how TLS relocations are emitted)
!
! First sequence uses LE_HIX22/LE_LOX10

! OBJDUMP: {{[0-9,a-f]+}}:  31 00 00 00  sethi 0, %i0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LE_HIX22 Head
! ASM: sethi %tle_hix22(Head), %i0 ! encoding: [0x31,0x00,0x00,0x00]
! ASM:                                 !   fixup A - offset: 0, value: %tle_hix22(Head), kind: fixup_sparc_tls_le_hix22
        sethi %tle_hix22(Head), %i0

! OBJDUMP: {{[0-9,a-f]+}}:  b0 1e 20 00  xor %i0, 0, %i0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LE_LOX10 Head
! ASM: xor %i0, %tle_lox10(Head), %i0 ! encoding: [0xb0,0x1e,0x20,0x00]
! ASM:                                    !   fixup A - offset: 0, value: %tle_lox10(Head), kind: fixup_sparc_tls_le_lox10
        xor %i0, %tle_lox10(Head), %i0


! Second sequence is for PIC, so it is more complicated.
! It uses LDO_HIX22/LDO_LOX10/LDO_ADD/LDM_HI22/LDM_LO10/LDM_ADD/LDM_CALL

! OBJDUMP: {{[0-9,a-f]+}}:  33 00 00 00  sethi 0, %i1
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDO_HIX22 Head
! ASM: sethi %tldo_hix22(Head), %i1 ! encoding: [0x33,0b00AAAAAA,A,A]
! ASM:                                  !   fixup A - offset: 0, value: %tldo_hix22(Head), kind: fixup_sparc_tls_ldo_hix22
        sethi %tldo_hix22(Head), %i1

! OBJDUMP: {{[0-9,a-f]+}}:  35 00 00 00  sethi 0, %i2
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDM_HI22 Head
! ASM: sethi %tldm_hi22(Head), %i2 ! encoding: [0x35,0b00AAAAAA,A,A]
! ASM:                                 !   fixup A - offset: 0, value: %tldm_hi22(Head), kind: fixup_sparc_tls_ldm_hi22
        sethi %tldm_hi22(Head), %i2

! OBJDUMP: {{[0-9,a-f]+}}:  b4 06 a0 00  add %i2, 0, %i2
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDM_LO10 Head
! ASM: add %i2, %tldm_lo10(Head), %i2 ! encoding: [0xb4,0x06,0b101000AA,A]
! ASM:                                    !   fixup A - offset: 0, value: %tldm_lo10(Head), kind: fixup_sparc_tls_ldm_lo10
        add %i2, %tldm_lo10(Head), %i2

	! ???error from llvm-mc on the next asm line???
	! add %i0, %i2, %o0, %tldm_add(Head)

! OBJDUMP: {{[0-9,a-f]+}}:  b0 1e 60 00  xor %i1, 0, %i0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDO_LOX10 Head
! ASM: xor %i1, %tldo_lox10(Head), %i0 ! encoding: [0xb0,0x1e,0b011000AA,A]
! ASM:                                     !   fixup A - offset: 0, value: %tldo_lox10(Head), kind: fixup_sparc_tls_ldo_lox10
        xor %i1, %tldo_lox10(Head), %i0

        ! ???error from llvm-mc on the next asm line???
        ! call __tls_get_addr, %tldm_call(Head)
        ! nop
        ! ???error from llvm-mc on the next asm line???
        ! add %o0, %i0, %i0, %tldo_add(Head)

        .type  Head,@object
        .section      .tbss,#alloc,#write,#tls
Head:
        .word  0
        .size  Head, 4
