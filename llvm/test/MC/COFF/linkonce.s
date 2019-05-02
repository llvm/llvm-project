// Test section manipulation via .linkonce directive.
//
// RUN: llvm-mc -triple i386-pc-win32 -filetype=obj %s | llvm-readobj -S --symbols | FileCheck %s
// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -S --symbols | FileCheck %s

.section s1
.linkonce
.long 1

.section s2
.linkonce one_only
.long 1

.section s3
.linkonce discard
.long 1

.section s4
.linkonce same_size
.long 1

.section s5
.linkonce same_contents
.long 1

.section s6
.long 1

.section s7
.linkonce largest
.long 1

.section s8
.linkonce newest
.long 1

.section .foo$bar
.linkonce discard
.long 1


// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Name: s1
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: s2
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: s3
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: s4
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: s5
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: s6
// CHECK:     Characteristics [
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: s7
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: s8
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: s1
// CHECK:     Section: s1 (4)
// CHECK:     AuxSectionDef {
// CHECK:       Number: 4
// CHECK:       Selection: Any (0x2)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: s2
// CHECK:     Section: s2 (5)
// CHECK:     AuxSectionDef {
// CHECK:       Number: 5
// CHECK:       Selection: NoDuplicates (0x1)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: s3
// CHECK:     Section: s3 (6)
// CHECK:     AuxSectionDef {
// CHECK:       Number: 6
// CHECK:       Selection: Any (0x2)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: s4
// CHECK:     Section: s4 (7)
// CHECK:     AuxSectionDef {
// CHECK:       Number: 7
// CHECK:       Selection: SameSize (0x3)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: s5
// CHECK:     Section: s5 (8)
// CHECK:     AuxSectionDef {
// CHECK:       Number: 8
// CHECK:       Selection: ExactMatch (0x4)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: s6
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: s7
// CHECK:     Section: s7 (10)
// CHECK:     AuxSectionDef {
// CHECK:       Number: 10
// CHECK:       Selection: Largest (0x6)
// CHECK:     }
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: s8
// CHECK:     Section: s8 (11)
// CHECK:     AuxSectionDef {
// CHECK:       Number: 11
// CHECK:       Selection: Newest (0x7)
// CHECK:     }
// CHECK:   }
