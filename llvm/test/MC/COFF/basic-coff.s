// This test checks that the COFF object emitter works for the most basic
// programs.

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
	movl	$L_.str, (%esp)
	calll	_printf
	xorl	%eax, %eax
	addl	$4, %esp
	ret

	.data
L_.str:                                 # @.str
	.asciz	 "Hello World"

// CHECK: {
// CHECK:   MachineType              = IMAGE_FILE_MACHINE_I386 (0x14C)
// CHECK:   NumberOfSections         = 2
// CHECK:   TimeDateStamp            = {{[0-9]+}}
// CHECK:   PointerToSymbolTable     = 0x{{[0-9A-F]+}}
// CHECK:   NumberOfSymbols          = 6
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
// CHECK:       NumberOfRelocations      = 2
// CHECK:       NumberOfLineNumbers      = 0
// CHECK:       Charateristics           = 0x60500020
// CHECK:         IMAGE_SCN_CNT_CODE
// CHECK:         IMAGE_SCN_ALIGN_16BYTES
// CHECK:         IMAGE_SCN_MEM_EXECUTE
// CHECK:         IMAGE_SCN_MEM_READ
// CHECK:       SectionData              =
// CHECK:       Relocations              = [
// CHECK:         0 = {
// CHECK:           VirtualAddress           = 0x{{[0-9A-F]+}}
// CHECK:           SymbolTableIndex         = 2
// CHECK:           Type                     = IMAGE_REL_I386_DIR32 (6)
// CHECK:           SymbolName               = .data
// CHECK:         }
// CHECK:         1 = {
// CHECK:           VirtualAddress           = 0x{{[0-9A-F]+}}
// CHECK:           SymbolTableIndex         = 5
// CHECK:           Type                     = IMAGE_REL_I386_REL32 (20)
// CHECK:           SymbolName               = _printf
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
// CHECK:       Charateristics           = 0xC0300040
// CHECK:         IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK:         IMAGE_SCN_ALIGN_4BYTES
// CHECK:         IMAGE_SCN_MEM_READ
// CHECK:         IMAGE_SCN_MEM_WRITE
// CHECK:       SectionData              =
// CHECK:         48 65 6C 6C 6F 20 57 6F - 72 6C 64 00             |Hello World.|
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
// CHECK:         15 00 00 00 02 00 00 00 - 00 00 00 00 01 00 00 00 |................|
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
// CHECK:         0C 00 00 00 00 00 00 00 - 00 00 00 00 02 00 00 00 |................|
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
// CHECK:     }
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
// CHECK:   ]
// CHECK: }
