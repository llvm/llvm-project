// RUN: llvm-mc -triple i386-pc-win32 -filetype=obj %s | llvm-readobj -s -t | FileCheck %s
// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -s -t | FileCheck %s

.section assocSec, "dr", discard, "assocSym"
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

.section SecName, "dr", associative, "assocSym"
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

.section assocSec, "dr", associative, "assocSym"
.globl Symbol8
Symbol8:
.long 1

// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Number: 4
// CHECK:     Name: assocSec
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 5
// CHECK:     Name: secName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 6
// CHECK:     Name: secName
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
// CHECK:   Section {
// CHECK:     Number: 9
// CHECK:     Name: SecName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 10
// CHECK:     Name: SecName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: 11
// CHECK:     Name: SecName
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: assocSec
// CHECK:     Section: assocSec (4)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Any
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: assocSym
// CHECK:     Section: assocSec
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: secName
// CHECK:     Section: secName (5)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Any
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol1
// CHECK:     Section: secName (5)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: secName
// CHECK:     Section: secName (6)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: NoDuplicates
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol2
// CHECK:     Section: secName (6)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: SecName
// CHECK:     Section: SecName (7)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: SameSize
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol3
// CHECK:     Section: SecName (7)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: SecName
// CHECK:     Section: SecName (8)
// CHECK:     AuxSymbolCount: 1
// CHECK:     AuxSectionDef {
// CHECK:       Selection: ExactMatch
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol4
// CHECK:     Section: SecName (8)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: SecName
// CHECK:     Section: SecName (11)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Associative
// CHECK:       AssocSection: assocSec (4)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: SecName
// CHECK:     Section: SecName (9)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Largest
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol6
// CHECK:     Section: SecName (9)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: SecName
// CHECK:     Section: SecName (10)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Newest (0x7)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol7
// CHECK:     Section: SecName (10)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: assocSec
// CHECK:     Section: assocSec (12)
// CHECK:     AuxSectionDef {
// CHECK:       Selection: Associative (0x5)
// CHECK:       AssocSection: assocSec (4)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol5
// CHECK:     Section: SecName (11)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: Symbol8
// CHECK:     Section: assocSec (12)
// CHECK:   }
// CHECK: ]
