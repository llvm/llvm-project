# RUN: llvm-mc -triple=mips-unknown-linux -filetype=obj %s -o - | \
# RUN:   llvm-readobj -mips-abi-flags | \
# RUN:   FileCheck --check-prefix=ASE-MICROMIPS %s

	.set	micromips
	.ent	_Z3foov
_Z3foov:
	addiu	$sp, $sp, -8

# ASE-MICROMIPS: microMIPS (0x800)
