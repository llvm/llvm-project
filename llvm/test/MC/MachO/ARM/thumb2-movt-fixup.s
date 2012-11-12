@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumbv7-apple-darwin10 -filetype=obj -o - < %s | macho-dump | FileCheck %s

_fred:
	movt	r3, :upper16:(_wilma-(LPC0_0+4))
LPC0_0:

_wilma:
  .long 0

@ CHECK:  ('_relocations', [
@ CHECK:    # Relocation 0
@ CHECK:    (('word-0', 0xb9000000),
@ CHECK:     ('word-1', 0x4)),
@ CHECK:    # Relocation 1
@ CHECK:    (('word-0', 0xb100fffc),
@ CHECK:     ('word-1', 0x4)),

