// This test checks that the COFF object emitter works for the most basic
// programs.

// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | llvm-readobj -h -s -sr -sd -t | FileCheck %s

.def	 _main;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	_main
	.align	16, 0x90
_main:                                  # @main
# %bb.0:                                # %entry
	subl	$4, %esp
	movl	$.L_.str, (%esp)
	call	_printf
	xorl	%eax, %eax
	addl	$4, %esp
	ret

	.data
.L_.str:                                # @.str
	.asciz	"Hello World"

// CHECK: ImageFileHeader {
// CHECK:   Machine: IMAGE_FILE_MACHINE_AMD64
// CHECK:   SectionCount: 3
// CHECK:   TimeDateStamp: {{[0-9]+}}
// CHECK:   PointerToSymbolTable: 0x{{[0-9A-F]+}}
// CHECK:   SymbolCount: 8
// CHECK:   OptionalHeaderSize: 0
// CHECK:   Characteristics [ (0x0)
// CHECK:   ]
// CHECK: }
// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Number:               [[TextNum:[0-9]+]]
// CHECK:     Name:                 .text
// CHECK:     VirtualSize:          0
// CHECK:     VirtualAddress:       0
// CHECK:     RawDataSize:          [[TextSize:[0-9]+]]
// CHECK:     PointerToRawData:     0x{{[0-9A-F]+}}
// CHECK:     PointerToRelocations: 0x{{[0-9A-F]+}}
// CHECK:     PointerToLineNumbers: 0x0
// CHECK:     RelocationCount:      2
// CHECK:     LineNumberCount:      0
// CHECK:     Characteristics [ (0x60500020)
// CHECK:       IMAGE_SCN_ALIGN_16BYTES
// CHECK:       IMAGE_SCN_CNT_CODE
// CHECK:       IMAGE_SCN_MEM_EXECUTE
// CHECK:       IMAGE_SCN_MEM_READ
// CHECK:     ]
// CHECK:     Relocations [
// CHECK:       0x{{[0-9A-F]+}} IMAGE_REL_AMD64_ADDR32 .data
// CHECK:       0x{{[0-9A-F]+}} IMAGE_REL_AMD64_REL32 _printf
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number:               [[DataNum:[0-9]+]]
// CHECK:     Name:                 .data
// CHECK:     VirtualSize:          0
// CHECK:     VirtualAddress:       0
// CHECK:     RawDataSize:          [[DataSize:[0-9]+]]
// CHECK:     PointerToRawData:     0x{{[0-9A-F]+}}
// CHECK:     PointerToRelocations: 0x0
// CHECK:     PointerToLineNumbers: 0x0
// CHECK:     RelocationCount:      0
// CHECK:     LineNumberCount:      0
// CHECK:     Characteristics [ (0xC0300040)
// CHECK:       IMAGE_SCN_ALIGN_4BYTES
// CHECK:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK:       IMAGE_SCN_MEM_READ
// CHECK:       IMAGE_SCN_MEM_WRITE
// CHECK:     ]
// CHECK:     Relocations [
// CHECK:     ]
// CHECK:     SectionData (
// CHECK:       0000: 48656C6C 6F20576F 726C6400             |Hello World.|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name:           .text
// CHECK:     Value:          0
// CHECK:     Section:        .text
// CHECK:     BaseType:       Null
// CHECK:     ComplexType:    Null
// CHECK:     StorageClass:   Static
// CHECK:     AuxSymbolCount: 1
// CHECK:     AuxSectionDef {
// CHECK:       Length: [[TextSize]]
// CHECK:       RelocationCount: 2
// CHECK:       LineNumberCount: 0
// CHECK:       Checksum: 0x8E1B6D20
// CHECK:       Number: [[TextNum]]
// CHECK:       Selection: 0x0
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name:           .data
// CHECK:     Value:          0
// CHECK:     Section:        .data
// CHECK:     BaseType:       Null
// CHECK:     ComplexType:    Null
// CHECK:     StorageClass:   Static
// CHECK:     AuxSymbolCount: 1
// CHECK:     AuxSectionDef {
// CHECK:       Length: [[DataSize]]
// CHECK:       RelocationCount: 0
// CHECK:       LineNumberCount: 0
// CHECK:       Checksum: 0x2B95CA92
// CHECK:       Number: [[DataNum]]
// CHECK:       Selection: 0x0
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name:           _main
// CHECK:     Value:          0
// CHECK:     Section:        .text
// CHECK:     BaseType:       Null
// CHECK:     ComplexType:    Function
// CHECK:     StorageClass:   External
// CHECK:     AuxSymbolCount: 0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name:           _printf
// CHECK:     Value:          0
// CHECK:     Section:        IMAGE_SYM_UNDEFINED (0)
// CHECK:     BaseType:       Null
// CHECK:     ComplexType:    Null
// CHECK:     StorageClass:   External
// CHECK:     AuxSymbolCount: 0
// CHECK:   }
// CHECK: ]
