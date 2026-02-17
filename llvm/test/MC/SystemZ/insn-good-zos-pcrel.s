* For z10 and above.
* RUN: llvm-mc -triple s390x-ibm-zos -show-encoding %s | FileCheck %s

*CHECK: * encoding: [0xc0,0x04,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: brcl	0, FOO
*CHECK: * encoding: [0xc0,0x04,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnop	FOO
	brcl	0,FOO
	jlnop	FOO

*CHECK: * encoding: [0xc0,0x84,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jge	FOO
*CHECK: * encoding: [0xc0,0x84,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jge	FOO
	jle	FOO
	brel	FOO

*CHECK: * encoding: [0xc0,0x74,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgne	FOO
*CHECK: * encoding: [0xc0,0x74,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgne	FOO
	jlne	FOO
	brnel	FOO

*CHECK: * encoding: [0xc0,0x24,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgh	FOO
*CHECK: * encoding: [0xc0,0x24,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgh	FOO
	jlh	FOO
	brhl	FOO

*CHECK: * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnh	FOO
*CHECK: * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnh	FOO
	jlnh	FOO
	brnhl	FOO

*CHECK: * encoding: [0xc0,0x44,A,A,A,A]
*CHECK: fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgl	FOO
*CHECK: * encoding: [0xc0,0x44,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgl	FOO
	jll	FOO
	brll	FOO

*CHECK: * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnl	FOO
*CHECK: * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnl	FOO
	jlnl	FOO
	brnll	FOO

*CHECK: * encoding: [0xc0,0x84,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgz	FOO
*CHECK: * encoding: [0xc0,0x84,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgz	FOO
	jlz	FOO
	brzl	FOO

*CHECK: * encoding: [0xc0,0x74,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnz	FOO
*CHECK: * encoding: [0xc0,0x74,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnz	FOO
	jlnz	FOO
	brnzl	FOO

*CHECK: * encoding: [0xc0,0x24,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgp	FOO
*CHECK: * encoding: [0xc0,0x24,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgp	FOO
	jlp	FOO
	brpl	FOO

*CHECK: * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnp	FOO
*CHECK: * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnp	FOO
	jlnp	FOO
	brnpl	FOO

*CHECK: * encoding: [0xc0,0x44,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgm	FOO
*CHECK: * encoding: [0xc0,0x44,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgm	FOO
	jlm	FOO
	brml	FOO


*CHECK: * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnm	FOO
*CHECK: * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jgnm	FOO
	jlnm	FOO
	brnml	FOO

*CHECK: * encoding: [0xc0,0xf4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jg	FOO
*CHECK: * encoding: [0xc0,0xf4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jg	FOO
	jlu	FOO
	brul	FOO

