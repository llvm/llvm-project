# RUN: llvm-mc --verify -triple riscv32 < %s
# RUN: llvm-mc --verify -triple riscv64 < %s

sf.cflush.d.l1 0x10 # expected-error:1 {{invalid instruction}}

sf.cdiscard.d.l1 0x10 # expected-error:1 {{invalid instruction}}

sf.cflush.d.l1 x0 # expected-error:1 {{instruction requires the following: 'XSiFivecflushdlone' (SiFive sf.cflush.d.l1 Instruction)}}

sf.cflush.d.l1 x7 # expected-error:1 {{instruction requires the following: 'XSiFivecflushdlone' (SiFive sf.cflush.d.l1 Instruction)}}

sf.cdiscard.d.l1 x0 # expected-error:1 {{instruction requires the following: 'XSiFivecdiscarddlone' (SiFive sf.cdiscard.d.l1 Instruction)}}

sf.cdiscard.d.l1 x7 # expected-error:1 {{instruction requires the following: 'XSiFivecdiscarddlone' (SiFive sf.cdiscard.d.l1 Instruction)}}

sf.cease x1 # expected-error:1 {{invalid instruction}}

sf.cease 0x10 # expected-error:1 {{invalid instruction}}

sf.cease # expected-error:1 {{instruction requires the following: 'XSfcease' (SiFive sf.cease Instruction)}}
