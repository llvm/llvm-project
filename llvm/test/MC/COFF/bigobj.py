# RUN: %python %s | llvm-mc -filetype=obj -triple i686-pc-win32 - | llvm-readobj -h - | FileCheck %s

from __future__ import print_function

# This test checks that the COFF object emitter can produce objects with
# more than 65279 sections.

# While we only generate 65277 sections, an implicit .text, .data and .bss will
# also be emitted.  This brings the total to 65280.
num_sections = 65277

# CHECK:      ImageFileHeader {
# CHECK-NEXT:   Machine: IMAGE_FILE_MACHINE_I386
# CHECK-NEXT:   SectionCount: 65280
# CHECK-NEXT:   TimeDateStamp: {{[0-9]+}}
# CHECK-NEXT:   PointerToSymbolTable: 0x{{[0-9A-F]+}}
# CHECK-NEXT:   SymbolCount: 195837
# CHECK-NEXT:   StringTableSize: {{[0-9]+}}
# CHECK-NEXT:   OptionalHeaderSize: 0
# CHECK-NEXT:   Characteristics [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT: }

for i in range(0, num_sections):
	print("""	.section	.bss,"bw",discard,_b%d
	.globl	_b%d                     # @b%d
_b%d:
	.byte	0                       # 0x0
""" % (i, i, i, i))
