// REQUIRES: asserts
// RUN: llvm-mc -triple x86_64-unknown-unknown  -debug-only=asm-matcher  %s 2>&1 | FileCheck %s

// CHECK: AsmMatcher: found 4 encodings with mnemonic 'pshufb'
// CHECK: Trying to match opcode MMX_PSHUFBrr
// CHECK:   Matching formal operand class MCK_VR64 against actual operand at index 1 (Memory: ModeSize=64,BaseReg=rip,Scale=1,Disp=CPI1_0): Opcode result: multiple operand mismatches, ignoring this opcode
// CHECK: Trying to match opcode PSHUFBrr
// CHECK:   Matching formal operand class MCK_FR16 against actual operand at index 1 (Memory: ModeSize=64,BaseReg=rip,Scale=1,Disp=CPI1_0): Opcode result: multiple operand mismatches, ignoring this opcode
// CHECK: Trying to match opcode PSHUFBrm
// CHECK:   Matching formal operand class MCK_Mem128 against actual operand at index 1 (Memory: ModeSize=64,BaseReg=rip,Scale=1,Disp=CPI1_0): match success using generic matcher
// CHECK:   Matching formal operand class MCK_FR16 against actual operand at index 2 (Reg:xmm1): match success using generic matcher
// CHECK:   Matching formal operand class InvalidMatchClass against actual operand at index 3: actual operand index out of range
// CHECK:   Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 2 encodings with mnemonic 'sha1rnds4'
// CHECK: Trying to match opcode SHA1RNDS4rri
// CHECK:   Matching formal operand class MCK_ImmUnsignedi8 against actual operand at index 1 (Imm:1): match success using generic matcher
// CHECK:   Matching formal operand class MCK_FR16 against actual operand at index 2 (Reg:xmm1): match success using generic matcher
// CHECK:   Matching formal operand class MCK_FR16 against actual operand at index 3 (Reg:xmm2): match success using generic matcher
// CHECK:   Matching formal operand class InvalidMatchClass against actual operand at index 4: actual operand index out of range
// CHECK:   Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 4 encodings with mnemonic 'pinsrw'
// CHECK: Trying to match opcode MMX_PINSRWrr
// CHECK:   Matching formal operand class MCK_ImmUnsignedi8 against actual operand at index 1 (Imm:3): match success using generic matcher
// CHECK:   Matching formal operand class MCK_GR32orGR64 against actual operand at index 2 (Reg:ecx): match success using generic matcher
// CHECK:   Matching formal operand class MCK_VR64 against actual operand at index 3 (Reg:xmm5): Opcode result: multiple operand mismatches, ignoring this opcode
// CHECK: Trying to match opcode PINSRWrr
// CHECK:   Matching formal operand class MCK_ImmUnsignedi8 against actual operand at index 1 (Imm:3): match success using generic matcher
// CHECK:   Matching formal operand class MCK_GR32orGR64 against actual operand at index 2 (Reg:ecx): match success using generic matcher
// CHECK:   Matching formal operand class MCK_FR16 against actual operand at index 3 (Reg:xmm5): match success using generic matcher
// CHECK:   Matching formal operand class InvalidMatchClass against actual operand at index 4: actual operand index out of range
// CHECK:   Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 2 encodings with mnemonic 'crc32l'
// CHECK: Trying to match opcode CRC32r32r32
// CHECK:   Matching formal operand class MCK_GR32 against actual operand at index 1 (Memory: ModeSize=64,BaseReg=rbx,IndexReg=rcx,Scale=8,Disp=3735928559,SegReg=gs): Opcode result: multiple operand mismatches, ignoring this opcode
// CHECK: Trying to match opcode CRC32r32m32
// CHECK:   Matching formal operand class MCK_Mem32 against actual operand at index 1 (Memory: ModeSize=64,BaseReg=rbx,IndexReg=rcx,Scale=8,Disp=3735928559,SegReg=gs): match success using generic matcher
// CHECK:   Matching formal operand class MCK_GR32 against actual operand at index 2 (Reg:ecx): match success using generic matcher
// CHECK:   Matching formal operand class InvalidMatchClass against actual operand at index 3: actual operand index out of range
// CHECK:   Opcode result: complete match, selecting this opcode
// CHECK: AsmMatcher: found 4 encodings with mnemonic 'punpcklbw'
// CHECK: Trying to match opcode MMX_PUNPCKLBWrr
// CHECK:   Matching formal operand class MCK_VR64 against actual operand at index 1 (Reg:mm0): match success using generic matcher
// CHECK:   Matching formal operand class MCK_VR64 against actual operand at index 2 (Memory: ModeSize=64,Size=32,BaseReg=rsp,Scale=1): Opcode result: multiple operand mismatches, ignoring this opcode
// CHECK: Trying to match opcode MMX_PUNPCKLBWrm
// CHECK:   Matching formal operand class MCK_VR64 against actual operand at index 1 (Reg:mm0): match success using generic matcher
// CHECK:   Matching formal operand class MCK_Mem32 against actual operand at index 2 (Memory: ModeSize=64,Size=32,BaseReg=rsp,Scale=1): match success using generic matcher
// CHECK:   Matching formal operand class InvalidMatchClass against actual operand at index 3: actual operand index out of range
// CHECK:   Opcode result: complete match, selecting this opcode

pshufb    CPI1_0(%rip), %xmm1
sha1rnds4 $1, %xmm1, %xmm2
pinsrw    $3, %ecx, %xmm5
crc32l    %gs:0xdeadbeef(%rbx,%rcx,8),%ecx

.intel_syntax
punpcklbw mm0, dword ptr [rsp]
