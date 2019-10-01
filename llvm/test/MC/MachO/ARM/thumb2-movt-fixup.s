@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumbv7-apple-darwin10 -filetype=obj -o - < %s | llvm-readobj -r --expand-relocs | FileCheck %s

_fred:
	movt	r3, :upper16:(_wilma-(LPC0_0+4))
LPC0_0:

_wilma:
  .long 0

@ CHECK: File: <stdin>
@ CHECK: Format: Mach-O arm
@ CHECK: Arch: arm
@ CHECK: AddressSize: 32bit
@ CHECK: Relocations [
@ CHECK:   Section __text {
@ CHECK:     Relocation {
@ CHECK:       Offset: 0x0
@ CHECK:       PCRel: 0
@ CHECK:       Length: 3
@ CHECK:       Type: ARM_RELOC_HALF_SECTDIFF (9)
@ CHECK:       Value: 0x4
@ CHECK:     }
@ CHECK:     Relocation {
@ CHECK:       Offset: 0xFFFC
@ CHECK:       PCRel: 0
@ CHECK:       Length: 3
@ CHECK:       Type: ARM_RELOC_PAIR (1)
@ CHECK:       Value: 0x4
@ CHECK:     }
@ CHECK:   }
@ CHECK: ]
