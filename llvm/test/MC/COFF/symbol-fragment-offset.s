// The purpose of this test is to see if the COFF object writer is emitting the
// proper relocations for multiple pieces of data in a single data fragment.

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj -h -s -sr -sd -t | FileCheck %s

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
	movl	$L_.str0, (%esp)
	calll	_printf
	movl	$L_.str1, (%esp)
	calll	_puts
	movl	$L_.str2, (%esp)
	calll	_puts
	xorl	%eax, %eax
	addl	$4, %esp
	ret

	.data
L_.str0:                                # @.str0
	.asciz	 "Hello "

L_.str1:                                # @.str1
	.asciz	 "World!"

	.align	16                      # @.str2
L_.str2:
	.asciz	 "I'm The Last Line."

// CHECK: {
// CHECK:   Machine:                   IMAGE_FILE_MACHINE_I386 (0x14C)
// CHECK:   SectionCount:              3
// CHECK:   TimeDateStamp:             {{[0-9]+}}
// CHECK:   PointerToSymbolTable:      0x{{[0-9A-F]+}}
// CHECK:   SymbolCount:               9
// CHECK:   OptionalHeaderSize:        0
// CHECK:   Characteristics [ (0x0)
// CHECK:   ]
// CHECK: }
// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Number:                    1
// CHECK:     Name:                      .text
// CHECK:     VirtualSize:               0
// CHECK:     VirtualAddress:            0
// CHECK:     RawDataSize:               {{[0-9]+}}
// CHECK:     PointerToRawData:          0x{{[0-9A-F]+}}
// CHECK:     PointerToRelocations:      0x{{[0-9A-F]+}}
// CHECK:     PointerToLineNumbers:      0x0
// CHECK:     RelocationCount:           6
// CHECK:     LineNumberCount:           0
// CHECK:     Characteristics [ (0x60500020)
// CHECK:       IMAGE_SCN_ALIGN_16BYTES
// CHECK:       IMAGE_SCN_CNT_CODE
// CHECK:       IMAGE_SCN_MEM_EXECUTE
// CHECK:       IMAGE_SCN_MEM_READ
// CHECK:     ]
// CHECK:     Relocations [
// CHECK:       0x6  IMAGE_REL_I386_DIR32 .data
// CHECK:       0xB  IMAGE_REL_I386_REL32 _printf
// CHECK:       0x12 IMAGE_REL_I386_DIR32 .data
// CHECK:       0x17 IMAGE_REL_I386_REL32 _puts
// CHECK:       0x1E IMAGE_REL_I386_DIR32 .data
// CHECK:       0x23 IMAGE_REL_I386_REL32 _puts
// CHECK:     ]
// CHECK:     SectionData (
// CHECK:       0000: 83EC04C7 04240000 0000E800 000000C7 |.....$..........|
// CHECK:       0010: 04240700 0000E800 000000C7 04241000 |.$...........$..|
// CHECK:       0020: 0000E800 00000031 C083C404 C3       |.......1.....|
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number:                    2
// CHECK:     Name:                      .data
// CHECK:     VirtualSize:               0
// CHECK:     VirtualAddress:            0
// CHECK:     RawDataSize:               {{[0-9]+}}
// CHECK:     PointerToRawData:          0x{{[0-9A-F]+}}
// CHECK:     PointerToRelocations:      0x0
// CHECK:     PointerToLineNumbers:      0x0
// CHECK:     RelocationCount:           0
// CHECK:     LineNumberCount:           0
// CHECK:     Characteristics [ (0xC0500040)
// CHECK:       IMAGE_SCN_ALIGN_16BYTES
// CHECK:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK:       IMAGE_SCN_MEM_READ
// CHECK:       IMAGE_SCN_MEM_WRITE
// CHECK:     Relocations [
// CHECK:     ]
// CHECK:     SectionData (
// CHECK:       0000: 48656C6C 6F200057 6F726C64 21000000 |Hello .World!...|
// CHECK:       0010: 49276D20 54686520 4C617374 204C696E |I'm The Last Lin|
// CHECK:       0020: 652E00                              |e..|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name:                      .text
// CHECK:     Value:                     0
// CHECK:     Section:                   .text
// CHECK:     BaseType:                  Null
// CHECK:     ComplexType:               Null
// CHECK:     StorageClass:              Static
// CHECK:     AuxSymbolCount:            1
// CHECK:     AuxSectionDef {
// CHECK:       Length: 45
// CHECK:       RelocationCount: 6
// CHECK:       LineNumberCount: 0
// CHECK:       Checksum: 0xDED1DC2
// CHECK:       Number: 1
// CHECK:       Selection: 0x0
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name:                      .data
// CHECK:     Value:                     0
// CHECK:     Section:                   .data
// CHECK:     BaseType:                  Null
// CHECK:     ComplexType:               Null
// CHECK:     StorageClass:              Static
// CHECK:     AuxSymbolCount:            1
// CHECK:     AuxSectionDef {
// CHECK:       Length: 35
// CHECK:       RelocationCount: 0
// CHECK:       LineNumberCount: 0
// CHECK:       Checksum: 0xB0A4C21
// CHECK:       Number: 2
// CHECK:       Selection: 0x0
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name:                      _main
// CHECK:     Value:                     0
// CHECK:     Section:                   .text
// CHECK:     BaseType:                  Null
// CHECK:     ComplexType:               Function
// CHECK:     StorageClass:              External
// CHECK:     AuxSymbolCount:            0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name:                      _printf
// CHECK:     Value:                     0
// CHECK:     Section:                   IMAGE_SYM_UNDEFINED (0)
// CHECK:     BaseType:                  Null
// CHECK:     ComplexType:               Null
// CHECK:     StorageClass:              External
// CHECK:     AuxSymbolCount:            0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name:                      _puts
// CHECK:     Value:                     0
// CHECK:     Section:                   IMAGE_SYM_UNDEFINED (0)
// CHECK:     BaseType:                  Null
// CHECK:     ComplexType:               Null
// CHECK:     StorageClass:              External
// CHECK:     AuxSymbolCount:            0
// CHECK:   }
// CHECK: ]
