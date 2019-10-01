// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj --file-headers -S -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols | FileCheck %s

_bar:
  nop
_foo:
  nop

  .set foo_set1, (_foo + 0xffff0000)
  .set foo_set2, (_foo - _bar + 0xffff0000)

foo_equals = (_foo + 0xffff0000)
foo_equals2 = (_foo - _bar + 0xffff0000)

  .globl foo_set1_global;
  .set foo_set1_global, (_foo + 0xffff0000)

  .globl foo_set2_global;
  .set foo_set2_global, (_foo - _bar + 0xffff0000)

// CHECK: File: <stdin>
// CHECK: Format: Mach-O 64-bit x86-64
// CHECK: Arch: x86_64
// CHECK: AddressSize: 64bit
// CHECK: MachHeader {
// CHECK:   Magic: Magic64 (0xFEEDFACF)
// CHECK:   CpuType: X86-64 (0x1000007)
// CHECK:   CpuSubType: CPU_SUBTYPE_X86_64_ALL (0x3)
// CHECK:   FileType: Relocatable (0x1)
// CHECK:   NumOfLoadCommands: 4
// CHECK:   SizeOfLoadCommands: 272
// CHECK:   Flags [ (0x0)
// CHECK:   ]
// CHECK:   Reserved: 0x0
// CHECK: }
// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Index: 0
// CHECK:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x2
// CHECK:     Offset: 304
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: 0x0
// CHECK:     Attributes [ (0x800004)
// CHECK:       PureInstructions (0x800000)
// CHECK:       SomeInstructions (0x4)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _bar (12)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _foo (17)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x1
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: foo_set1 (75)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x220)
// CHECK:       AltEntry (0x200)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0xFFFF0001
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: foo_set2 (54)
// CHECK:     Type: Abs (0x2)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0xFFFF0001
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: foo_equals (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x200)
// CHECK:       AltEntry (0x200)
// CHECK:     ]
// CHECK:     Value: 0xFFFF0001
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: foo_equals2 (63)
// CHECK:     Type: Abs (0x2)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0xFFFF0001
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: foo_set1_global (38)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x220)
// CHECK:	AltEntry (0x200)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0xFFFF0001
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: foo_set2_global (22)
// CHECK:     Extern
// CHECK:     Type: Abs (0x2)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0xFFFF0001
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
// CHECK:   Size: 152
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x2
// CHECK:   fileoff: 304
// CHECK:   filesize: 2
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 1
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 6
// CHECK:   iextdefsym: 6
// CHECK:   nextdefsym: 2
// CHECK:   iundefsym: 8
// CHECK:   nundefsym: 0
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
