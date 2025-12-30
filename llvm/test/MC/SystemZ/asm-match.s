// REQUIRES: asserts
// RUN: llvm-mc -triple s390x-linux-gnu -debug-only=asm-matcher %s 2>&1 | FileCheck %s
//
// Check that debug output prints the operands correctly.

// CHECK: AsmMatcher: found 1 encodings with mnemonic 'sllg'
// CHECK: Trying to match opcode SLLG
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 1 (Reg:r3): match success using generic matcher
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 2 (Reg:r0): match success using generic matcher
// CHECK: Matching formal operand class MCK_BDAddr32Disp20 against actual operand at index 3 (Mem:3): match success using generic matcher
// CHECK: Matching formal operand class InvalidMatchClass against actual operand at index 4: actual operand index out of range
// CHECK: Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 1 encodings with mnemonic 'llill'
// CHECK: Trying to match opcode LLILL
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 1 (Reg:r0): match success using generic matcher
// CHECK: Matching formal operand class MCK_U16Imm against actual operand at index 2 (Imm:0): match success using generic matcher
// CHECK: Matching formal operand class InvalidMatchClass against actual operand at index 3: actual operand index out of range
// CHECK: Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 1 encodings with mnemonic 'lgr'
// CHECK: Trying to match opcode LGR
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 1 (Reg:r1): match success using generic matcher
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 2 (Reg:r0): match success using generic matcher
// CHECK: Matching formal operand class InvalidMatchClass against actual operand at index 3: actual operand index out of range
// CHECK: Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 1 encodings with mnemonic 'lg'
// CHECK: Trying to match opcode LG
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 1 (Reg:r1): match success using generic matcher
// CHECK: Matching formal operand class MCK_BDXAddr64Disp20 against actual operand at index 2 (Mem:16(r2)): match success using generic matcher
// CHECK: Matching formal operand class InvalidMatchClass against actual operand at index 3: actual operand index out of range
// CHECK: Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 1 encodings with mnemonic 'lg'
// CHECK: Trying to match opcode LG
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 1 (Reg:r1): match success using generic matcher
// CHECK: Matching formal operand class MCK_BDXAddr64Disp20 against actual operand at index 2 (Mem:16(r2,r3)): match success using generic matcher
// CHECK: Matching formal operand class InvalidMatchClass against actual operand at index 3: actual operand index out of range
// CHECK: Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 1 encodings with mnemonic 'stmg'
// CHECK: Trying to match opcode STMG
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 1 (Reg:r13): match success using generic matcher
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 2 (Reg:r15): match success using generic matcher
// CHECK: Matching formal operand class MCK_BDAddr64Disp20 against actual operand at index 3 (Mem:104(r15)): match success using generic matcher
// CHECK: Matching formal operand class InvalidMatchClass against actual operand at index 4: actual operand index out of range
// CHECK: Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 1 encodings with mnemonic 'mvc'
// CHECK: Trying to match opcode MVC
// CHECK: Matching formal operand class MCK_BDLAddr64Disp12Len8 against actual operand at index 1 (Mem:184(8,r15)): match success using generic matcher
// CHECK: Matching formal operand class MCK_BDAddr64Disp12 against actual operand at index 2 (Mem:8(r2)): match success using generic matcher
// CHECK: Matching formal operand class InvalidMatchClass against actual operand at index 3: actual operand index out of range
// CHECK: Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 1 encodings with mnemonic 'mvck'
// CHECK: Trying to match opcode MVCK
// CHECK: Matching formal operand class MCK_BDRAddr64Disp12 against actual operand at index 1 (Mem:0(r0,r1)): match success using generic matcher
// CHECK: Matching formal operand class MCK_BDAddr64Disp12 against actual operand at index 2 (Mem:4095(r15)): match success using generic matcher
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 3 (Reg:r2): match success using generic matcher
// CHECK: Matching formal operand class InvalidMatchClass against actual operand at index 4: actual operand index out of range
// CHECK: Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 1 encodings with mnemonic 'j'
// CHECK: Trying to match opcode J
// CHECK: Matching formal operand class MCK_PCRel16 against actual operand at index 1 (Imm:.Ltmp0+2): match success using generic matcher
// CHECK: Matching formal operand class InvalidMatchClass against actual operand at index 2: actual operand index out of range
// CHECK: Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 1 encodings with mnemonic 'brasl'
// CHECK: Trying to match opcode BRASL
// CHECK: Matching formal operand class MCK_GR64 against actual operand at index 1 (Reg:r14): match success using generic matcher
// CHECK: Matching formal operand class MCK_PCRelTLS32 against actual operand at index 2 (ImmTLS:fun): match success using generic matcher
// CHECK: Matching formal operand class InvalidMatchClass against actual operand at index 3: actual operand index out of range
// CHECK: Opcode result: complete match, selecting this opcode
// CHECK: sllg	%r3, %r0, 3
// CHECK: llill	%r0, 0
// CHECK: lgr	%r1, %r0
// CHECK: lg	%r1, 16(%r2)
// CHECK: lg	%r1, 16(%r2,%r3)
// CHECK: stmg	%r13, %r15, 104(%r15)
// CHECK: mvc	184(8,%r15), 8(%r2)
// CHECK: mvck	0(%r0,%r1), 4095(%r15), %r2
// CHECK: .Ltmp0:
// CHECK: j	.Ltmp0+2
// CHECK: brasl	%r14, fun
	
        sllg    %r3, %r0, 3
        llill	%r0, 0
        lgr	%r1, %r0
        lg      %r1, 16(%r2)
        lg      %r1, 16(%r2,%r3)
        stmg    %r13, %r15, 104(%r15)
        mvc     184(8,%r15), 8(%r2)
        mvck    0(%r0,%r1), 4095(%r15), %r2
.Ltmp0:
        j	.Ltmp0+2
        brasl   %r14, fun
