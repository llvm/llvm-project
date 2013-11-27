// RUN: llvm-mc -triple i386-pc-win32 -filetype=obj %s | llvm-readobj -s -t | FileCheck %s
// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -s -t | FileCheck %s

.section assocSec
.linkonce
.long 1

.section secName, "dr", discard, "Symbol1"
.globl Symbol1
Symbol1:
.long 1

.section secName, "dr", one_only, "Symbol2"
.globl Symbol2
Symbol2:
.long 1

.section SecName, "dr", same_size, "Symbol3"
.globl Symbol3
Symbol3:
.long 1

.section SecName, "dr", same_contents, "Symbol4"
.globl Symbol4
Symbol4:
.long 1

.section SecName, "dr", associative assocSec, "Symbol5"
.globl Symbol5
Symbol5:
.long 1

.section SecName, "dr", largest, "Symbol6"
.globl Symbol6
Symbol6:
.long 1

.section SecName, "dr", newest, "Symbol7"
.globl Symbol7
Symbol7:
.long 1

// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Number: 1
// CHECK:     Name: assocSec
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 2
// CHECK:     Name: secName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 3
// CHECK:     Name: secName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 4
// CHECK:     Name: SecName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 5
// CHECK:     Name: SecName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 6
// CHECK:     Name: SecName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 7
// CHECK:     Name: SecName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 8
// CHECK:     Name: SecName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: assocSec
// CHECK:     Section: assocSec (1)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Any
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: secName
// CHECK:     Section: secName (2)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Any
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: secName
// CHECK:     Section: secName (3)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: NoDuplicates
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: SecName
// CHECK:     Section: SecName (4)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: SameSize
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: SecName
// CHECK:     Section: SecName (5)
// CHECK:     AuxSymbolCount: 1
// CHECK:     AuxSectionDef {
// CHECK:       Selection: ExactMatch
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: SecName
// CHECK:     Section: SecName (6)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Associative
// CHECK:       AssocSection: assocSec (1)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: SecName
// CHECK:     Section: SecName (7)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Largest
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: SecName
// CHECK:     Section: SecName (8)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Newest (0x7)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol1
// CHECK:     Section: secName (2)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol2
// CHECK:     Section: secName (3)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol3
// CHECK:     Section: SecName (4)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol4
// CHECK:     Section: SecName (5)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol5
// CHECK:     Section: SecName (6)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol6
// CHECK:     Section: SecName (7)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol7
// CHECK:     Section: SecName (8)
// CHECK:   }
// CHECK: ]
