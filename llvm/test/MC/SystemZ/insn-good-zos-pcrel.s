* For z10 and above.
* RUN: llvm-mc -triple s390x-ibm-zos -show-encoding %s | FileCheck %s

*CHECK: * encoding: [0xc0,0x04,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: brcl	0,FOO
*CHECK: * encoding: [0xc0,0x04,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnop	FOO
	brcl	0,FOO
	jlnop	FOO

*CHECK: * encoding: [0xc0,0x84,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jle	FOO
*CHECK: * encoding: [0xc0,0x84,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jle	FOO
	jle	FOO
	brel	FOO

*CHECK: * encoding: [0xc0,0x74,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlne	FOO
*CHECK: * encoding: [0xc0,0x74,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlne	FOO
	jlne	FOO
	brnel	FOO

*CHECK: * encoding: [0xc0,0x24,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlh	FOO
*CHECK: * encoding: [0xc0,0x24,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlh	FOO
	jlh	FOO
	brhl	FOO

*CHECK: * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnh	FOO
*CHECK: * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnh	FOO
	jlnh	FOO
	brnhl	FOO

*CHECK: * encoding: [0xc0,0x44,A,A,A,A]
*CHECK: fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jll	FOO
*CHECK: * encoding: [0xc0,0x44,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jll	FOO
	jll	FOO
	brll	FOO

*CHECK: * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnl	FOO
*CHECK: * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnl	FOO
	jlnl	FOO
	brnll	FOO

*CHECK: * encoding: [0xc0,0x84,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlz	FOO
*CHECK: * encoding: [0xc0,0x84,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlz	FOO
	jlz	FOO
	brzl	FOO

*CHECK: * encoding: [0xc0,0x74,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnz	FOO
*CHECK: * encoding: [0xc0,0x74,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnz	FOO
	jlnz	FOO
	brnzl	FOO

*CHECK: * encoding: [0xc0,0x24,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlp	FOO
*CHECK: * encoding: [0xc0,0x24,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlp	FOO
	jlp	FOO
	brpl	FOO

*CHECK: * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnp	FOO
*CHECK: * encoding: [0xc0,0xd4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnp	FOO
	jlnp	FOO
	brnpl	FOO

*CHECK: * encoding: [0xc0,0x44,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlm	FOO
*CHECK: * encoding: [0xc0,0x44,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlm	FOO
	jlm	FOO
	brml	FOO


*CHECK: * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnm	FOO
*CHECK: * encoding: [0xc0,0xb4,A,A,A,A]
*CHECK:  fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlnm	FOO
	jlnm	FOO
	brnml	FOO

*CHECK: * encoding: [0xc0,0xf4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlu	FOO
*CHECK: * encoding: [0xc0,0xf4,A,A,A,A]
*CHECK: * fixup A - offset: 2, value: FOO+2, kind: FK_390_PC32DBL
*CHECK: jlu	FOO
	jlu	FOO
	brul	FOO

