* For z10 and above.
* RUN: llvm-mc -triple s390x-ibm-zos -show-encoding %s | FileCheck %s

*CHECK: brcl	0, FOO                  * encoding: [0xc0,0x04,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: brcl	0, FOO                  * encoding: [0xc0,0x04,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	brcl	0,FOO
	jlnop	FOO

*CHECK: jge	FOO                     * encoding: [0xc0,0x84,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jge	FOO                     * encoding: [0xc0,0x84,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jle	FOO
	brel	FOO

*CHECK: jgne	FOO                     * encoding: [0xc0,0x74,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgne	FOO                     * encoding: [0xc0,0x74,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlne	FOO
	brnel	FOO

*CHECK: jgh	FOO                     * encoding: [0xc0,0x24,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgh	FOO                     * encoding: [0xc0,0x24,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlh	FOO
	brhl	FOO

*CHECK: jgnh	FOO                     * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnh	FOO                     * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlnh	FOO
	brnhl	FOO

*CHECK: jgl	FOO                     * encoding: [0xc0,0x44,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgl	FOO                     * encoding: [0xc0,0x44,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jll	FOO
	brll	FOO

*CHECK: jgnl	FOO                     * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnl	FOO                     * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlnl	FOO
	brnll	FOO

*CHECK: jgz	FOO                     * encoding: [0xc0,0x84,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgz	FOO                     * encoding: [0xc0,0x84,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlz	FOO
	brzl	FOO

*CHECK: jgnz	FOO                     * encoding: [0xc0,0x74,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnz	FOO                     * encoding: [0xc0,0x74,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlnz	FOO
	brnzl	FOO

*CHECK: jgp	FOO                     * encoding: [0xc0,0x24,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgp	FOO                     * encoding: [0xc0,0x24,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlp	FOO
	brpl	FOO

*CHECK: jgnp	FOO                     * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnp	FOO                     * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlnp	FOO
	brnpl	FOO

*CHECK: jgm	FOO                     * encoding: [0xc0,0x44,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgm	FOO                     * encoding: [0xc0,0x44,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlm	FOO
	brml	FOO


*CHECK: jgnm	FOO                     * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnm	FOO                     * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlnm	FOO
	brnml	FOO

*CHECK: jg	FOO                     * encoding: [0xc0,0xf4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jg	FOO                     * encoding: [0xc0,0xf4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
	jlu	FOO
	brul	FOO

