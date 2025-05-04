# RUN: llvm-mc %s -triple=sparc | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc %s -triple=sparcv9 | FileCheck %s --check-prefix=ASM

# RUN: llvm-mc %s -triple=sparc -filetype=obj -o %t
# RUN: llvm-objdump -dr %t | FileCheck %s --check-prefix=OBJDUMP
# RUN: llvm-mc %s -triple=sparcv9 -filetype=obj -o %t
# RUN: llvm-objdump -dr %t | FileCheck %s --check-prefix=OBJDUMP
# RUN: llvm-readelf -s - < %t | FileCheck %s --check-prefix=READELF --implicit-check-not=TLS

# READELF: TLS     LOCAL  DEFAULT [[#]] s_tle_hix22
# READELF: TLS     LOCAL  DEFAULT [[#]] s_tldo_hix22
# READELF: TLS     GLOBAL DEFAULT   UND s_tle_lox10
# READELF: TLS     GLOBAL DEFAULT   UND s_tie_hi22
# READELF: TLS     GLOBAL DEFAULT   UND s_tie_lo10
# READELF: TLS     GLOBAL DEFAULT   UND s_tie_ld
# READELF: TLS     GLOBAL DEFAULT   UND s_tie_ldx
# READELF: TLS     GLOBAL DEFAULT   UND s_tie_add
# READELF: TLS     GLOBAL DEFAULT   UND s_tldm_hi22
# READELF: TLS     GLOBAL DEFAULT   UND s_tldm_lo10
# READELF: TLS     GLOBAL DEFAULT   UND s_tldm_add
# READELF: TLS     GLOBAL DEFAULT   UND s_tldo_lox10
# READELF: TLS     GLOBAL DEFAULT   UND s_tldo_add
# READELF: TLS     GLOBAL DEFAULT   UND s_tgd_hi22
# READELF: TLS     GLOBAL DEFAULT   UND s_tgd_lo10
# READELF: TLS     GLOBAL DEFAULT   UND s_tgd_add

# ASM:      or %g1, %lo(sym), %g3
# ASM-NEXT: sethi %hi(sym), %l0
# ASM-NEXT: sethi %h44(sym), %l0
# ASM-NEXT: or %g1, %m44(sym), %g3
# ASM-NEXT: or %g1, %l44(sym), %g3
# OBJDUMP:     0000000:  R_SPARC_LO10	sym
# OBJDUMP:     0000004:  R_SPARC_HI22	sym
# OBJDUMP:     0000008:  R_SPARC_H44	sym
# OBJDUMP:     000000c:  R_SPARC_M44	sym
# OBJDUMP:     0000010:  R_SPARC_L44	sym
or %g1, %lo(sym), %g3
sethi %hi(sym), %l0
sethi %h44(sym), %l0
or %g1, %m44(sym), %g3
or %g1, %l44(sym), %g3

# ASM:      sethi %hh(sym), %l0
# ASM-NEXT: sethi %hh(sym), %l0
# ASM-NEXT: or %g1, %hm(sym), %g3
# ASM-NEXT: or %g1, %hm(sym), %g3
# ASM-NEXT: sethi %lm(sym), %l0
# OBJDUMP:     0000014:  R_SPARC_HH22	sym
# OBJDUMP:     0000018:  R_SPARC_HH22	sym
# OBJDUMP:     000001c:  R_SPARC_HM10	sym
# OBJDUMP:     0000020:  R_SPARC_HM10	sym
# OBJDUMP:     0000024:  R_SPARC_LM22	sym
sethi %hh(sym), %l0
sethi %uhi(sym), %l0
or %g1, %hm(sym), %g3
or %g1, %ulo(sym), %g3
sethi %lm(sym), %l0

# ASM:      sethi %hix(sym), %g1
# ASM-NEXT: xor %g1, %lox(sym), %g1
# ASM-NEXT: sethi %gdop_hix22(sym), %l1
# ASM-NEXT: or %l1, %gdop_lox10(sym), %l1
# ASM-NEXT: ldx [%l7+%l1], %l2, %gdop(sym)
# OBJDUMP: R_SPARC_HIX22 sym
# OBJDUMP: R_SPARC_LOX10 sym
# OBJDUMP: R_SPARC_GOTDATA_HIX22 sym
# OBJDUMP: R_SPARC_GOTDATA_LOX10 sym
# OBJDUMP: R_SPARC_GOTDATA_OP sym
sethi %hix(sym), %g1
xor %g1, %lox(sym), %g1
sethi %gdop_hix22(sym), %l1
or %l1, %gdop_lox10(sym), %l1
ldx [%l7 + %l1], %l2, %gdop(sym)

# OBJDUMP-LABEL: <.tls>:
.section .tls,"ax"
## Local Executable model:
# ASM:      sethi %tle_hix22(s_tle_hix22), %i0
# ASM-NEXT: xor %i0, %tle_lox10(s_tle_lox10), %i0

# OBJDUMP:      31 00 00 00   sethi 0x0, %i0
# OBJDUMP-NEXT:  00000000:  R_SPARC_TLS_LE_HIX22 s_tle_hix22
# OBJDUMP-NEXT: b0 1e 20 00   xor %i0, 0x0, %i0
# OBJDUMP-NEXT:  00000004:  R_SPARC_TLS_LE_LOX10 s_tle_lox10
        sethi %tle_hix22(s_tle_hix22), %i0
        xor %i0, %tle_lox10(s_tle_lox10), %i0

## Initial Executable model
# ASM:      sethi %tie_hi22(s_tie_hi22), %i1
# ASM-NEXT: add %i1, %tie_lo10(s_tie_lo10), %i1
# ASM-NEXT: ld [%i0+%i1], %i0, %tie_ld(s_tie_ld)
# ASM-NEXT: ldx [%i0+%i1], %i0, %tie_ldx(s_tie_ldx)
# ASM-NEXT: add %g7, %i0, %o0, %tie_add(s_tie_add)

# OBJDUMP:      R_SPARC_TLS_IE_HI22	s_tie_hi22
# OBJDUMP:      R_SPARC_TLS_IE_LO10	s_tie_lo10
# OBJDUMP:      R_SPARC_TLS_IE_LD	s_tie_ld
# OBJDUMP:      R_SPARC_TLS_IE_LDX	s_tie_ldx
# OBJDUMP:      R_SPARC_TLS_IE_ADD	s_tie_add
	sethi %tie_hi22(s_tie_hi22), %i1
        add %i1, %tie_lo10(s_tie_lo10), %i1
        ld [%i0+%i1], %i0, %tie_ld(s_tie_ld)
        ldx [%i0+%i1], %i0, %tie_ldx(s_tie_ldx)
        add %g7, %i0, %o0, %tie_add(s_tie_add)

## Local Dynamic model
# ASM:      sethi %tldo_hix22(s_tldo_hix22), %i1
# ASM-NEXT: sethi %tldm_hi22(s_tldm_hi22), %i2
# ASM-NEXT: add %i2, %tldm_lo10(s_tldm_lo10), %i2
# ASM-NEXT: add %i0, %i2, %o0, %tldm_add(s_tldm_add)
# ASM-NEXT: xor %i1, %tldo_lox10(s_tldo_lox10), %i0
# ASM-NEXT: call __tls_get_addr, %tldm_call(s_tldm_call)
# ASM-NEXT: nop
# ASM-NEXT: add %o0, %i0, %o0, %tldo_add(s_tldo_add)

# OBJDUMP:      R_SPARC_TLS_LDO_HIX22	s_tldo_hix22
# OBJDUMP:      R_SPARC_TLS_LDM_HI22	s_tldm_hi22
# OBJDUMP:      R_SPARC_TLS_LDM_LO10	s_tldm_lo10
# OBJDUMP:      R_SPARC_TLS_LDM_ADD	s_tldm_add
# OBJDUMP:      R_SPARC_TLS_LDO_LOX10	s_tldo_lox10
# OBJDUMP:      R_SPARC_TLS_LDM_CALL	s_tldm_call
# OBJDUMP:      R_SPARC_TLS_LDO_ADD	s_tldo_add
        sethi %tldo_hix22(s_tldo_hix22), %i1
        sethi %tldm_hi22(s_tldm_hi22), %i2
        add %i2, %tldm_lo10(s_tldm_lo10), %i2
	add %i0, %i2, %o0, %tldm_add(s_tldm_add)
        xor %i1, %tldo_lox10(s_tldo_lox10), %i0
        call __tls_get_addr, %tldm_call(s_tldm_call)
        nop
        add %o0, %i0, %o0, %tldo_add(s_tldo_add)

## General Dynamic model
# ASM:      sethi %tgd_hi22(s_tgd_hi22), %i1
# ASM-NEXT: add %i1, %tgd_lo10(s_tgd_lo10), %i1
# ASM-NEXT: add %i0, %i1, %o0, %tgd_add(s_tgd_add)
# ASM-NEXT: call __tls_get_addr, %tgd_call(s_tgd_call)

# OBJDUMP:      R_SPARC_TLS_GD_HI22	s_tgd_hi22
# OBJDUMP:      R_SPARC_TLS_GD_LO10	s_tgd_lo10
# OBJDUMP:      R_SPARC_TLS_GD_ADD	s_tgd_add
# OBJDUMP:      R_SPARC_TLS_GD_CALL	s_tgd_call
        sethi %tgd_hi22(s_tgd_hi22), %i1
        add %i1, %tgd_lo10(s_tgd_lo10), %i1
        add %i0, %i1, %o0, %tgd_add(s_tgd_add)
        call __tls_get_addr, %tgd_call(s_tgd_call)

        .type  Local,@object
        .section      .tbss,#alloc,#write,#tls
s_tle_hix22:
s_tldo_hix22:
        .word  0
        .size  Local, 4
