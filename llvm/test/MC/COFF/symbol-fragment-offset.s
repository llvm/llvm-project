// The purpose of this test is to see if the COFF object writer is emitting the
// proper relocations for multiple pieces of data in a single data fragment.

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | coff-dump.py | FileCheck %s
// I WOULD RUN, BUT THIS FAILS: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s

.def	 _main;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	_main
	.align	16, 0x90
_main:                                  # @main
# BB#0:                                 # %entry
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
// CHECK:   MachineType              = IMAGE_FILE_MACHINE_I386 (0x14C)
// CHECK:   NumberOfSections         = 2
// CHECK:   TimeDateStamp            = {{[0-9]+}}
// CHECK:   PointerToSymbolTable     = 0x{{[0-9A-F]+}}
// CHECK:   NumberOfSymbols          = 7
// CHECK:   SizeOfOptionalHeader     = 0
// CHECK:   Characteristics          = 0x0
// CHECK:   Sections                 = [
// CHECK:     1 = {
// CHECK:       Name                     = .text
// CHECK:       VirtualSize              = 0
// CHECK:       VirtualAddress           = 0
// CHECK:       SizeOfRawData            = {{[0-9]+}}
// CHECK:       PointerToRawData         = 0x{{[0-9A-F]+}}
// CHECK:       PointerToRelocations     = 0x{{[0-9A-F]+}}
// CHECK:       PointerToLineNumbers     = 0x0
// CHECK:       NumberOfRelocations      = 6
// CHECK:       NumberOfLineNumbers      = 0
// CHECK:       Charateristics           = 0x60500020
// CHECK:         IMAGE_SCN_CNT_CODE
// CHECK:         IMAGE_SCN_ALIGN_16BYTES
// CHECK:         IMAGE_SCN_MEM_EXECUTE
// CHECK:         IMAGE_SCN_MEM_READ
// CHECK:       SectionData              =
// CHECK:         83 EC 04 C7 04 24 00 00 - 00 00 E8 00 00 00 00 C7 |.....$..........|
// CHECK:         04 24 07 00 00 00 E8 00 - 00 00 00 C7 04 24 10 00 |.$...........$..|
// CHECK:         00 00 E8 00 00 00 00 31 - C0 83 C4 04 C3 |.......1.....|
// CHECK:       Relocations              = [
// CHECK:         0 = {
// CHECK:           VirtualAddress           = 0x6
// CHECK:           SymbolTableIndex         = 2
// CHECK:           Type                     = IMAGE_REL_I386_DIR32 (6)
// CHECK:           SymbolName               = .data
// CHECK:         }
// CHECK:         1 = {
// CHECK:           VirtualAddress           = 0xB
// CHECK:           SymbolTableIndex         = 5
// CHECK:           Type                     = IMAGE_REL_I386_REL32 (20)
// CHECK:           SymbolName               = _printf
// CHECK:         }
// CHECK:         2 = {
// CHECK:           VirtualAddress           = 0x12
// CHECK:           SymbolTableIndex         = 2
// CHECK:           Type                     = IMAGE_REL_I386_DIR32 (6)
// CHECK:           SymbolName               = .data
// CHECK:         }
// CHECK:         3 = {
// CHECK:           VirtualAddress           = 0x17
// CHECK:           SymbolTableIndex         = 6
// CHECK:           Type                     = IMAGE_REL_I386_REL32 (20)
// CHECK:           SymbolName               = _puts
// CHECK:         }
// CHECK:         4 = {
// CHECK:           VirtualAddress           = 0x1E
// CHECK:           SymbolTableIndex         = 2
// CHECK:           Type                     = IMAGE_REL_I386_DIR32 (6)
// CHECK:           SymbolName               = .data
// CHECK:         }
// CHECK:         5 = {
// CHECK:           VirtualAddress           = 0x23
// CHECK:           SymbolTableIndex         = 6
// CHECK:           Type                     = IMAGE_REL_I386_REL32 (20)
// CHECK:           SymbolName               = _puts
// CHECK:         }
// CHECK:       ]
// CHECK:     }
// CHECK:     2 = {
// CHECK:       Name                     = .data
// CHECK:       VirtualSize              = 0
// CHECK:       VirtualAddress           = 0
// CHECK:       SizeOfRawData            = {{[0-9]+}}
// CHECK:       PointerToRawData         = 0x{{[0-9A-F]+}}
// CHECK:       PointerToRelocations     = 0x0
// CHECK:       PointerToLineNumbers     = 0x0
// CHECK:       NumberOfRelocations      = 0
// CHECK:       NumberOfLineNumbers      = 0
// CHECK:       Charateristics           = 0xC0500040
// CHECK:         IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK:         IMAGE_SCN_ALIGN_16BYTES
// CHECK:         IMAGE_SCN_MEM_READ
// CHECK:         IMAGE_SCN_MEM_WRITE
// CHECK:       SectionData              =
// CHECK:         48 65 6C 6C 6F 20 00 57 - 6F 72 6C 64 21 00 00 00 |Hello .World!...|
// CHECK:         49 27 6D 20 54 68 65 20 - 4C 61 73 74 20 4C 69 6E |I'm The Last Lin|
// CHECK:         65 2E 00                                          |e..|
// CHECK:       Relocations              = None
// CHECK:     }
// CHECK:   ]
// CHECK:   Symbols                  = [
// CHECK:     0 = {
// CHECK:       Name                     = .text
// CHECK:       Value                    = 0
// CHECK:       SectionNumber            = 1
// CHECK:       SimpleType               = IMAGE_SYM_TYPE_NULL (0)
// CHECK:       ComplexType              = IMAGE_SYM_DTYPE_NULL (0)
// CHECK:       StorageClass             = IMAGE_SYM_CLASS_STATIC (3)
// CHECK:       NumberOfAuxSymbols       = 1
// CHECK:       AuxillaryData            =
// CHECK:         2D 00 00 00 06 00 00 00 - 00 00 00 00 01 00 00 00 |-...............|
// CHECK:         00 00                                             |..|

// CHECK:     }
// CHECK:     2 = {
// CHECK:       Name                     = .data
// CHECK:       Value                    = 0
// CHECK:       SectionNumber            = 2
// CHECK:       SimpleType               = IMAGE_SYM_TYPE_NULL (0)
// CHECK:       ComplexType              = IMAGE_SYM_DTYPE_NULL (0)
// CHECK:       StorageClass             = IMAGE_SYM_CLASS_STATIC (3)
// CHECK:       NumberOfAuxSymbols       = 1
// CHECK:       AuxillaryData            =
// CHECK:         23 00 00 00 00 00 00 00 - 00 00 00 00 02 00 00 00 |#...............|
// CHECK:         00 00                                             |..|

// CHECK:     }
// CHECK:     4 = {
// CHECK:       Name                     = _main
// CHECK:       Value                    = 0
// CHECK:       SectionNumber            = 1
// CHECK:       SimpleType               = IMAGE_SYM_TYPE_NULL (0)
// CHECK:       ComplexType              = IMAGE_SYM_DTYPE_FUNCTION (2)
// CHECK:       StorageClass             = IMAGE_SYM_CLASS_EXTERNAL (2)
// CHECK:       NumberOfAuxSymbols       = 0
// CHECK:       AuxillaryData            =

// CHECK:     5 = {
// CHECK:       Name                     = _printf
// CHECK:       Value                    = 0
// CHECK:       SectionNumber            = 0
// CHECK:       SimpleType               = IMAGE_SYM_TYPE_NULL (0)
// CHECK:       ComplexType              = IMAGE_SYM_DTYPE_NULL (0)
// CHECK:       StorageClass             = IMAGE_SYM_CLASS_EXTERNAL (2)
// CHECK:       NumberOfAuxSymbols       = 0
// CHECK:       AuxillaryData            =

// CHECK:     }
// CHECK:     6 = {
// CHECK:       Name                     = _puts
// CHECK:       Value                    = 0
// CHECK:       SectionNumber            = 0
// CHECK:       SimpleType               = IMAGE_SYM_TYPE_NULL (0)
// CHECK:       ComplexType              = IMAGE_SYM_DTYPE_NULL (0)
// CHECK:       StorageClass             = IMAGE_SYM_CLASS_EXTERNAL (2)
// CHECK:       NumberOfAuxSymbols       = 0
// CHECK:       AuxillaryData            =

// CHECK:     }
// CHECK:   ]
// CHECK: }
