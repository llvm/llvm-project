// RUN: llvm-mc -filetype=obj -triple mips-unknown-linux -mattr=+micromips < %s \
// RUN:     | llvm-readobj --sections --section-data - \
// RUN:     | FileCheck -check-prefix=DATA %s
//
// XFAIL: *

// Please merge this with relocation.s when it passes.

// baz is equivalent to .text+0x8 and is recorded in the symbol table as such
// but it refers to microMIPS code so the addend must indicate this in the LSB.
// The addend must therefore be 0x9.
// DATA-LABEL: Name: .text
// DATA:       SectionData (
// DATA:       0000: 30430000 30420009 30430000 30420009
        addiu $2, $3, %got(baz)
        addiu $2, $2, %lo(baz)
baz:
        addiu $2, $3, %hi(baz)
        addiu $2, $2, %lo(baz)

// DATA-LABEL: Name: .data
// DATA:       SectionData (
	.data
	.word 0
bar:
	.word 1
// baz is equivalent to .text+0x8 and is recorded in the symbol table as such
// but it refers to microMIPS code so the addend must indicate this in the LSB.
// The addend must therefore be 0x9.
// DATA:       0000: 00000000 00000001 00000009
        .gpword baz
