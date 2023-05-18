! Testing Sparc TLS relocations emission
! (for now a couple local ones).
!
! RUN: llvm-mc %s -triple=sparc -show-encoding | FileCheck %s --check-prefix=ASM
! RUN: llvm-mc %s -triple=sparcv9 -show-encoding | FileCheck %s --check-prefix=ASM
! RUN: llvm-mc %s -triple=sparc -filetype=obj | llvm-readobj -r - | FileCheck %s --check-prefix=REL
! RUN: llvm-mc %s -triple=sparcv9 -filetype=obj | llvm-readobj -r - | FileCheck %s --check-prefix=REL
! RUN: llvm-mc %s -triple=sparc -filetype=obj | llvm-objdump -r -d - | FileCheck %s --check-prefix=OBJDUMP
! RUN: llvm-mc %s -triple=sparcv9 -filetype=obj | llvm-objdump -r -d - | FileCheck %s --check-prefix=OBJDUMP

! REL: Arch: sparc
! REL: Relocations [
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LE_HIX22 Local 0x0
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LE_LOX10 Local 0x0
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LDO_HIX22 Local 0x0
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LDM_HI22  Local 0x0
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LDM_LO10  Local 0x0
! REL: 0x{{[0-9,A-F]+}} R_SPARC_TLS_LDO_LOX10 Local 0x0
! REL: ]


! OBJDUMP: <foo>:
foo:
! Here we use two different sequences to get the address of a static TLS variable 'Local'
! (note - there is no intent to have valid assembler function here,
!  we just check how TLS relocations are emitted)
!
! Sequence for Local Executable model:
!     LE_HIX22/LE_LOX10

! OBJDUMP: {{[0-9,a-f]+}}:  31 00 00 00  sethi 0, %i0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LE_HIX22 Local
! ASM: sethi %tle_hix22(Local), %i0 ! encoding: [0x31,0x00,0x00,0x00]
! ASM:                             !   fixup A - offset: 0, value: %tle_hix22(Local), kind: fixup_sparc_tls_le_hix22
        sethi %tle_hix22(Local), %i0

! OBJDUMP: {{[0-9,a-f]+}}:  b0 1e 20 00  xor %i0, 0, %i0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LE_LOX10 Local
! ASM: xor %i0, %tle_lox10(Local), %i0 ! encoding: [0xb0,0x1e,0x20,0x00]
! ASM:                                !   fixup A - offset: 0, value: %tle_lox10(Local), kind: fixup_sparc_tls_le_lox10
        xor %i0, %tle_lox10(Local), %i0


! Second sequence is for PIC, so it is more complicated.
! Local Dynamic model:
!     LDO_HIX22/LDO_LOX10/LDO_ADD/LDM_HI22/LDM_LO10/LDM_ADD/LDM_CALL

! OBJDUMP: {{[0-9,a-f]+}}:  33 00 00 00  sethi 0, %i1
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDO_HIX22 Local
! ASM: sethi %tldo_hix22(Local), %i1 ! encoding: [0x33,0b00AAAAAA,A,A]
! ASM:                              !   fixup A - offset: 0, value: %tldo_hix22(Local), kind: fixup_sparc_tls_ldo_hix22
        sethi %tldo_hix22(Local), %i1

! OBJDUMP: {{[0-9,a-f]+}}:  35 00 00 00  sethi 0, %i2
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDM_HI22 Local
! ASM: sethi %tldm_hi22(Local), %i2 ! encoding: [0x35,0b00AAAAAA,A,A]
! ASM:                             !   fixup A - offset: 0, value: %tldm_hi22(Local), kind: fixup_sparc_tls_ldm_hi22
        sethi %tldm_hi22(Local), %i2

! OBJDUMP: {{[0-9,a-f]+}}:  b4 06 a0 00  add %i2, 0, %i2
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDM_LO10 Local
! ASM: add %i2, %tldm_lo10(Local), %i2 ! encoding: [0xb4,0x06,0b101000AA,A]
! ASM:                                !   fixup A - offset: 0, value: %tldm_lo10(Local), kind: fixup_sparc_tls_ldm_lo10
        add %i2, %tldm_lo10(Local), %i2

! OBJDUMP: {{[0-9,a-f]+}}:  90 06 00 1a add %i0, %i2, %o0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDM_ADD Local
! ASM: add %i0, %i2, %o0, %tldm_add(Local) ! encoding: [0x90,0x06,0x00,0x1a]
! ASM:                                    !   fixup A - offset: 0, value: %tldm_add(Local), kind: fixup_sparc_tls_ldm_add
	add %i0, %i2, %o0, %tldm_add(Local)

! OBJDUMP: {{[0-9,a-f]+}}:  b0 1e 60 00  xor %i1, 0, %i0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDO_LOX10 Local
! ASM: xor %i1, %tldo_lox10(Local), %i0 ! encoding: [0xb0,0x1e,0b011000AA,A]
! ASM:                                 !   fixup A - offset: 0, value: %tldo_lox10(Local), kind: fixup_sparc_tls_ldo_lox10
        xor %i1, %tldo_lox10(Local), %i0

! OBJDUMP: {{[0-9,a-f]+}}:  40 00 00 00 call 0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDM_CALL Local
! ASM: call __tls_get_addr, %tldm_call(Local) ! encoding: [0x40,0x00,0x00,0x00]
! ASM:                                       !   fixup A - offset: 0, value: %tldm_call(Local), kind: fixup_sparc_tls_ldm_call
        call __tls_get_addr, %tldm_call(Local)
        nop

! OBJDUMP: {{[0-9,a-f]+}}:  90 02 00 18 add %o0, %i0, %o0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_LDO_ADD Local
! ASM: add %o0, %i0, %o0, %tldo_add(Local) ! encoding: [0x90,0x02,0x00,0x18]
! ASM:                                    !   fixup A - offset: 0, value: %tldo_add(Local), kind: fixup_sparc_tls_ldo_add
        add %o0, %i0, %o0, %tldo_add(Local)

! Next two sequences are for extern symbols.
! Initial Executable model:
!     IE_HI22/IE_LO10/IE_LD (or IE_LDX)/IE_ADD

! OBJDUMP: {{[0-9,a-f]+}}:  33 00 00 00  sethi 0, %i1
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_IE_HI22 Extern
! ASM: sethi %tie_hi22(Extern), %i1 ! encoding: [0x33,0b00AAAAAA,A,A]
! ASM:                                    !   fixup A - offset: 0, value: %tie_hi22(Extern), kind: fixup_sparc_tls_ie_hi22
	sethi %tie_hi22(Extern), %i1

! OBJDUMP: {{[0-9,a-f]+}}:  b2 06 60 00  add %i1, 0, %i1
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_IE_LO10 Extern
! ASM: add %i1, %tie_lo10(Extern), %i1 ! encoding: [0xb2,0x06,0b011000AA,A]
! ASM:                                    !   fixup A - offset: 0, value: %tie_lo10(Extern), kind: fixup_sparc_tls_ie_lo10
        add %i1, %tie_lo10(Extern), %i1

! OBJDUMP: {{[0-9,a-f]+}}:  f0 06 00 19  ld [%i0+%i1], %i0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_IE_LD Extern
! ASM: ld [%i0+%i1], %i0, %tie_ld(Extern) ! encoding: [0xf0,0x06,0x00,0x19]
! ASM:                                    !   fixup A - offset: 0, value: %tie_ld(Extern), kind: fixup_sparc_tls_ie_ld
        ld [%i0+%i1], %i0, %tie_ld(Extern)

! OBJDUMP: {{[0-9,a-f]+}}:  f0 5e 00 19  ldx [%i0+%i1], %i0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_IE_LDX Extern
! ASM: ldx [%i0+%i1], %i0, %tie_ldx(Extern) ! encoding: [0xf0,0x5e,0x00,0x19]
! ASM:                                      !   fixup A - offset: 0, value: %tie_ldx(Extern), kind: fixup_sparc_tls_ie_ldx
        ldx [%i0+%i1], %i0, %tie_ldx(Extern)

! OBJDUMP: {{[0-9,a-f]+}}:  90 01 c0 18  add %g7, %i0, %o0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_IE_ADD Extern
! ASM: add %g7, %i0, %o0, %tie_add(Extern) ! encoding: [0x90,0x01,0xc0,0x18]
! ASM:                                    !   fixup A - offset: 0, value: %tie_add(Extern), kind: fixup_sparc_tls_ie_add
        add %g7, %i0, %o0, %tie_add(Extern)

! General Dynamic model
!     GD_HI22/GD_LO10/GD_ADD/GD_CALL

! OBJDUMP: {{[0-9,a-f]+}}:  33 00 00 00  sethi 0, %i1
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_GD_HI22 Extern
! ASM:  sethi %tgd_hi22(Extern), %i1    ! encoding: [0x33,0b00AAAAAA,A,A]
! ASM:                                    !   fixup A - offset: 0, value: %tgd_hi22(Extern), kind: fixup_sparc_tls_gd_hi22
        sethi %tgd_hi22(Extern), %i1

! OBJDUMP: {{[0-9,a-f]+}}:  b2 06 60 00  add %i1, 0, %i1
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_GD_LO10 Extern
! ASM: add %i1, %tgd_lo10(Extern), %i1 ! encoding: [0xb2,0x06,0b011000AA,A]
! ASM:                                 !   fixup A - offset: 0, value: %tgd_lo10(Extern), kind: fixup_sparc_tls_gd_lo10
        add %i1, %tgd_lo10(Extern), %i1

! OBJDUMP: {{[0-9,a-f]+}}:  90 06 00 19  add %i0, %i1, %o0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_GD_ADD Extern
! ASM: add %i0, %i1, %o0, %tgd_add(Extern) ! encoding: [0x90,0x06,0x00,0x19]
! ASM:                                    !   fixup A - offset: 0, value: %tgd_add(Extern), kind: fixup_sparc_tls_gd_add
        add %i0, %i1, %o0, %tgd_add(Extern)

! OBJDUMP: {{[0-9,a-f]+}}:  40 00 00 00 call 0
! OBJDUMP: {{[0-9,a-f]+}}:     R_SPARC_TLS_GD_CALL Extern
! ASM: call __tls_get_addr, %tgd_call(Extern) ! encoding: [0x40,0x00,0x00,0x00]
! ASM:                                        !   fixup A - offset: 0, value: %tgd_call(Extern), kind: fixup_sparc_tls_gd_call
        call __tls_get_addr, %tgd_call(Extern)

        .type  Local,@object
        .section      .tbss,#alloc,#write,#tls
Local:
        .word  0
        .size  Local, 4
