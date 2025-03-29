// RUN: llvm-mc -triple=arm64-apple-darwin < %s | \
// RUN:   FileCheck %s --check-prefix=ASM

// RUN: llvm-mc -triple=arm64-apple-darwin -filetype=obj < %s | \
// RUN:   llvm-readobj --expand-relocs --sections \
// RUN:   --section-relocations --section-data - | \
// RUN:   FileCheck %s --check-prefix=RELOC



// RELOC:    Sections [
// RELOC-LABEL: Section {
// RELOC-LABEL: Section {
// RELOC-NEXT:   Index: 1
// RELOC-NEXT:   Name: __const (5F 5F 63 6F 6E 73 74 00 00 00 00 00 00 00 00 00)
// RELOC-NEXT:   Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)

.section	__DATA,__const
.p2align	3

// RELOC-LABEL: Relocations [
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x70
// RELOC-NEXT:     PCRel: 0
// RELOC-NEXT:     Length: 3
// RELOC-NEXT:     Type: ARM64_RELOC_AUTHENTICATED_POINTER (11)
// RELOC-NEXT:     Symbol: _g 7
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x60
// RELOC-NEXT:     PCRel: 0
// RELOC-NEXT:     Length: 3
// RELOC-NEXT:     Type: ARM64_RELOC_AUTHENTICATED_POINTER (11)
// RELOC-NEXT:     Symbol: _g 6
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x50
// RELOC-NEXT:     PCRel: 0
// RELOC-NEXT:     Length: 3
// RELOC-NEXT:     Type: ARM64_RELOC_AUTHENTICATED_POINTER (11)
// RELOC-NEXT:     Symbol: _g5
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x40
// RELOC-NEXT:     PCRel: 0
// RELOC-NEXT:     Length: 3
// RELOC-NEXT:     Type: ARM64_RELOC_AUTHENTICATED_POINTER (11)
// RELOC-NEXT:     Symbol: _g4
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x30
// RELOC-NEXT:     PCRel: 0
// RELOC-NEXT:     Length: 3
// RELOC-NEXT:     Type: ARM64_RELOC_AUTHENTICATED_POINTER (11)
// RELOC-NEXT:     Symbol: _g3
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x20
// RELOC-NEXT:     PCRel: 0
// RELOC-NEXT:     Length: 3
// RELOC-NEXT:     Type: ARM64_RELOC_AUTHENTICATED_POINTER (11)
// RELOC-NEXT:     Symbol: _g2
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x10
// RELOC-NEXT:     PCRel: 0
// RELOC-NEXT:     Length: 3
// RELOC-NEXT:     Type: ARM64_RELOC_AUTHENTICATED_POINTER (11)
// RELOC-NEXT:     Symbol: _g1
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x0
// RELOC-NEXT:     PCRel: 0
// RELOC-NEXT:     Length: 3
// RELOC-NEXT:     Type: ARM64_RELOC_AUTHENTICATED_POINTER (11)
// RELOC-NEXT:     Symbol: _g0
// RELOC-NEXT:   }
// RELOC-NEXT: ]
// RELOC-NEXT: SectionData (

// RELOC-NEXT:   0000: 00000000 2A000080
// ASM:          .quad _g0@AUTH(ia,42)
.quad _g0@AUTH(ia,42)
.quad 0

// RELOC-NEXT:   0010: 00000000 00000280
// ASM:          .quad _g1@AUTH(ib,0)
.quad _g1@AUTH(ib,0)
.quad 0

// RELOC-NEXT:   0020: 00000000 05000580
// ASM:          .quad _g2@AUTH(da,5,addr)
.quad _g2@AUTH(da,5,addr)
.quad 0

// RELOC-NEXT:   0030: 00000000 FFFF0780
// ASM:          .quad _g3@AUTH(db,65535,addr)
.quad _g3@AUTH(db,0xffff,addr)
.quad 0

// RELOC-NEXT:   0040: 07000000 00000080
// ASM:          .quad (_g4+7)@AUTH(ia,0)
.quad (_g4 + 7)@AUTH(ia,0)
.quad 0

// RELOC-NEXT:   0050: FDFFFFFF 00DE0280
// ASM:          .quad (_g5-3)@AUTH(ib,56832)
.quad (_g5 - 3)@AUTH(ib,0xde00)
.quad 0

// RELOC-NEXT:   0060: 00000000 FF000780
// ASM:          .quad "_g 6"@AUTH(db,255,addr)
.quad "_g 6"@AUTH(db,0xff,addr)
.quad 0

// RELOC-NEXT:   0070: 07000000 10000080
// ASM:          .quad ("_g 7"+7)@AUTH(ia,16)
.quad ("_g 7" + 7)@AUTH(ia,16)
.quad 0
