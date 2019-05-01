// The purpose of this test is to see if the COFF object writer is emitting the
// proper relocations for multiple pieces of data in a single data fragment.

// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | llvm-readobj -h -S --sr --sd --symbols | FileCheck %s

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
	movl	$.L_.str0, (%esp)
	callq	_printf
	movl	$.L_.str1, (%esp)
	callq	_puts
	movl	$.L_.str2, (%esp)
	callq	_puts
	xorl	%eax, %eax
	addl	$4, %esp
	ret

	.data
.L_.str0:                                # @.str0
	.asciz	 "Hello "

.L_.str1:                                # @.str1
	.asciz	 "World!"

	.align	16                      # @.str2
.L_.str2:
	.asciz	 "I'm The Last Line."

// CHECK: {
// CHECK:   Machine:                   IMAGE_FILE_MACHINE_AMD64
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
// CHECK:       0x7  IMAGE_REL_AMD64_ADDR32 .data
// CHECK:       0xC  IMAGE_REL_AMD64_REL32 _printf
// CHECK:       0x14 IMAGE_REL_AMD64_ADDR32 .data
// CHECK:       0x19 IMAGE_REL_AMD64_REL32 _puts
// CHECK:       0x21 IMAGE_REL_AMD64_ADDR32 .data
// CHECK:       0x26 IMAGE_REL_AMD64_REL32 _puts
// CHECK:     ]
// CHECK:     SectionData (
// CHECK:       0000: 83EC0467 C7042400 000000E8 00000000
// CHECK:       0010: 67C70424 07000000 E8000000 0067C704
// CHECK:       0020: 24100000 00E80000 000031C0 83C404C3
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
// CHECK:       Length: 48
// CHECK:       RelocationCount: 6
// CHECK:       LineNumberCount: 0
// CHECK:       Checksum: 0x7BD396E3
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
