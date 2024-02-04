# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s --check-prefixes=CHECK
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 | \
# RUN:   FileCheck %s --check-prefixes=CHECK

	.text
	.abicalls
	.option	pic2
xx:
	la	$2,.Lhello #CHECK: lw      $2, %got(.Lhello)($gp)          # encoding: [0x8f,0x82,A,A]
                           #CHECK: #   fixup A - offset: 0, value: %got(.Lhello), kind: fixup_Mips_GOT
                           #CHECK: addiu   $2, $2, %lo(.Lhello)            # encoding: [0x24,0x42,A,A]
                           #CHECK: #   fixup A - offset: 0, value: %lo(.Lhello), kind: fixup_Mips_LO16

	la	$2,$hello2 #CHECK: lw      $2, %got($hello2)($gp)          # encoding: [0x8f,0x82,A,A]
                           #CHECK: #   fixup A - offset: 0, value: %got($hello2), kind: fixup_Mips_GOT
                           #CHECK: addiu   $2, $2, %lo($hello2)            # encoding: [0x24,0x42,A,A]
                           #CHECK: #   fixup A - offset: 0, value: %lo($hello2), kind: fixup_Mips_LO16
	.rdata
.Lhello:
	.asciz "Hello world\n"
$hello2:
	.asciz "Hello world\n"
