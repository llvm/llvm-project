// RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - | FileCheck %s

        .section        __TEXT,__text,regular,pure_instructions
        .section        __DATA,__thread_data,thread_local_regular
        .globl  _c$tlv$init
        .align  2
_c$tlv$init:
        .long   4

        .section        __DATA,__thread_vars,thread_local_variables
        .globl  _c
_c:
        .quad   ___tlv_bootstrap
        .quad   0
        .quad   _c$tlv$init

        .section        __DATA,__thread_data,thread_local_regular
        .globl  _d$tlv$init
        .align  2
_d$tlv$init:
        .long   5

        .section        __DATA,__thread_vars,thread_local_variables
        .globl  _d
_d:
        .quad   ___tlv_bootstrap
        .quad   0
        .quad   _d$tlv$init

.tbss _a$tlv$init, 4, 2

        .globl  _a
_a:
        .quad   ___tlv_bootstrap
        .quad   0
        .quad   _a$tlv$init

.tbss _b$tlv$init, 4, 2

        .globl  _b
_b:
        .quad   ___tlv_bootstrap
        .quad   0
        .quad   _b$tlv$init

.subsections_via_symbols

// CHECK: File: <stdin>
// CHECK: Format: Mach-O 64-bit x86-64
// CHECK: Arch: x86_64
// CHECK: AddressSize: 64bit
// CHECK: MachHeader {
// CHECK:   Magic: Magic64 (0xFEEDFACF)
// CHECK:   CpuType: X86-64 (0x1000007)
// CHECK:   CpuSubType: CPU_SUBTYPE_X86_64_ALL (0x3)
// CHECK:   FileType: Relocatable (0x1)
// CHECK:   NumOfLoadCommands: 3
// CHECK:   SizeOfLoadCommands: 496
// CHECK:   Flags [ (0x2000)
// CHECK:     MH_SUBSECTIONS_VIA_SYMBOLS (0x2000)
// CHECK:   ]
// CHECK:   Reserved: 0x0
// CHECK: }
// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Index: 0
// CHECK:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 528
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x800000)
// CHECK:       PureInstructions (0x800000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 1
// CHECK:     Name: __thread_data (5F 5F 74 68 72 65 61 64 5F 64 61 74 61 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x8
// CHECK:     Offset: 528
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: ThreadLocalRegular (0x11)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 04000000 05000000                    |........|
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 2
// CHECK:     Name: __thread_vars (5F 5F 74 68 72 65 61 64 5F 76 61 72 73 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x8
// CHECK:     Size: 0x60
// CHECK:     Offset: 536
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x278
// CHECK:     RelocationCount: 8
// CHECK:     Type: ThreadLocalVariables (0x13)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 00000000 00000000 00000000 00000000  |................|
// CHECK:       0010: 00000000 00000000 00000000 00000000  |................|
// CHECK:       0020: 00000000 00000000 00000000 00000000  |................|
// CHECK:       0030: 00000000 00000000 00000000 00000000  |................|
// CHECK:       0040: 00000000 00000000 00000000 00000000  |................|
// CHECK:       0050: 00000000 00000000 00000000 00000000  |................|
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 3
// CHECK:     Name: __thread_bss (5F 5F 74 68 72 65 61 64 5F 62 73 73 00 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x68
// CHECK:     Size: 0x8
// CHECK:     Offset: 0
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: ThreadLocalZerofill (0x12)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: CFFAEDFE 07000001                    |........|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __thread_vars {
// CHECK:     0x58 0 3 1 X86_64_RELOC_UNSIGNED 0 _b$tlv$init
// CHECK:     0x48 0 3 1 X86_64_RELOC_UNSIGNED 0 ___tlv_bootstrap
// CHECK:     0x40 0 3 1 X86_64_RELOC_UNSIGNED 0 _a$tlv$init
// CHECK:     0x30 0 3 1 X86_64_RELOC_UNSIGNED 0 ___tlv_bootstrap
// CHECK:     0x28 0 3 1 X86_64_RELOC_UNSIGNED 0 _d$tlv$init
// CHECK:     0x18 0 3 1 X86_64_RELOC_UNSIGNED 0 ___tlv_bootstrap
// CHECK:     0x10 0 3 1 X86_64_RELOC_UNSIGNED 0 _c$tlv$init
// CHECK:     0x0 0 3 1 X86_64_RELOC_UNSIGNED 0 ___tlv_bootstrap
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _a$tlv$init (37)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __thread_bss (0x4)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x68
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _b$tlv$init (25)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __thread_bss (0x4)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x6C
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _a (75)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __thread_vars (0x3)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x38
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _b (72)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __thread_vars (0x3)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x50
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _c (69)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __thread_vars (0x3)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x8
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _c$tlv$init (13)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __thread_data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _d (66)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __thread_vars (0x3)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x20
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _d$tlv$init (1)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __thread_data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x4
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: ___tlv_bootstrap (49)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK: ]
// CHECK: Indirect Symbols {
// CHECK:   Number: 0
// CHECK:   Symbols [
// CHECK:   ]
// CHECK: }
// CHECK: Segment {
// CHECK:   Cmd: LC_SEGMENT_64
// CHECK:   Name: 
// CHECK:   Size: 392
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x70
// CHECK:   fileoff: 528
// CHECK:   filesize: 104
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 4
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 2
// CHECK:   iextdefsym: 2
// CHECK:   nextdefsym: 6
// CHECK:   iundefsym: 8
// CHECK:   nundefsym: 1
// CHECK:   tocoff: 0
// CHECK:   ntoc: 0
// CHECK:   modtaboff: 0
// CHECK:   nmodtab: 0
// CHECK:   extrefsymoff: 0
// CHECK:   nextrefsyms: 0
// CHECK:   indirectsymoff: 0
// CHECK:   nindirectsyms: 0
// CHECK:   extreloff: 0
// CHECK:   nextrel: 0
// CHECK:   locreloff: 0
// CHECK:   nlocrel: 0
// CHECK: }
