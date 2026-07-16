; RUN: llvm-mc -triple riscv32-apple-macho %s -o %t.o -filetype=obj
; RUN: llvm-objdump -dr %t.o | FileCheck %s
; RUN: llvm-otool -Vtr %t.o | FileCheck %s --check-prefix=OTOOL

Lpcrel_hi0:
	auipc	a0, %pcrel_hi(_glob)
	.rep 511
	nop
	.endr
	lw	a1, %pcrel_lo(Lpcrel_hi0)(a0)
	addi	a1, a1, 1
	;; 4(auipc) + 511 * 4(nop) + 4(lw) + 4(addi) = 4 + 2044 + 8 = 2056
        ;; -2056 is 0xfff7f8 in 24-bit two's complement representation
	sw	a1, %pcrel_lo(Lpcrel_hi0)(a0)
	ret

; CHECK-LABEL: 00000000 <ltmp0>:
; CHECK-LABEL:   0: 00000517     	auipc	a0, 0x0
; CHECK-NEXT:  		                00000000:  RISCV_RELOC_HI20(pcrel)	_glob
; CHECK-NEXT:    4: 00000013     	nop
; CHECK-NEXT:    8: 00000013     	nop
; CHECK-NEXT:    c: 00000013     	nop
; ...
; CHECK-LABEL: 7fc: 00000013     	nop
; CHECK-NEXT:  800: 80052583     	lw	a1, -0x800(a0)
; CHECK-NEXT: 	          		00000800:  RISCV_RELOC_LO12(pcrel)	_glob
; CHECK-NEXT:  804: 00158593     	addi	a1, a1, 0x1
; CHECK-NEXT:  808: 00b52023     	sw	a1, 0x0(a0)
; CHECK-NEXT:            		00000808:  RISCV_RELOC_ADDEND	0xfff7f8
; CHECK-NEXT:            		00000808:  RISCV_RELOC_LO12(pcrel)	_glob
; CHECK-NEXT:  80c: 00008067     	ret

; OTOOL-LABEL: Relocation information (__TEXT,__text) 4 entries
; OTOOL-NEXT:  address  pcrel length extern type    scattered symbolnum/value
; OTOOL-NEXT:  00000808 False long   False  8       False     addend = 0xfff7f8
; OTOOL-NEXT:  00000808 True  long   True   4       False     _glob
; OTOOL-NEXT:  00000800 True  long   True   4       False     _glob
; OTOOL-NEXT:  00000000 True  long   True   3       False     _glob
; OTOOL-NEXT:  Contents of (__TEXT,__text) section
; OTOOL-NEXT:  00000000        auipc   a0, 0x0
; ...
; OTOOL-LABEL: 000007fc        nop
; OTOOL-NEXT:  00000800        lw      a1, -0x800(a0)
; OTOOL-NEXT:  00000804        addi    a1, a1, 0x1
; OTOOL-NEXT:  00000808        sw      a1, 0x0(a0)
; OTOOL-NEXT:  0000080c        ret
