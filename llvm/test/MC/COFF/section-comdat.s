// RUN: llvm-mc -triple i386-pc-win32 -filetype=obj %s | llvm-objdump -h -t - | FileCheck %s
// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-objdump -h -t - | FileCheck %s

.section assocSec, "dr", discard, "assocSym"
.global assocSym
assocSym:
.long assocSec

.section secName, "dr", discard, "Symbol1"
.globl Symbol1
Symbol1:
.long assocSym

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

# CHECK:      Sections:
# CHECK-NEXT: Idx Name          Size
# CHECK-NEXT:   0 .text         00000000
# CHECK-NEXT:   1 .data         00000000
# CHECK-NEXT:   2 .bss          00000000
# CHECK-NEXT:   3 assocSec      00000004
# CHECK-NEXT:   4 secName       00000004
# CHECK-NEXT:   5 secName       00000004
# CHECK-NEXT:   6 SecName       00000004
# CHECK-NEXT:   7 SecName       00000004
# CHECK-NEXT:   8 SecName       00000004
# CHECK-NEXT:   9 SecName       00000004
# CHECK-NEXT:  10 SecName       00000004
# CHECK-NEXT:  11 assocSec      00000004
# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: [ 0](sec  1)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .text
# CHECK-NEXT: AUX scnlen 0x0 nreloc 0 nlnno 0 checksum 0x0 assoc 1 comdat 0
# CHECK-NEXT: [ 2](sec  2)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .data
# CHECK-NEXT: AUX scnlen 0x0 nreloc 0 nlnno 0 checksum 0x0 assoc 2 comdat 0
# CHECK-NEXT: [ 4](sec  3)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .bss
# CHECK-NEXT: AUX scnlen 0x0 nreloc 0 nlnno 0 checksum 0x0 assoc 3 comdat 0
# CHECK-NEXT: [ 6](sec  4)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 assocSec
# CHECK-NEXT: AUX scnlen 0x4 nreloc 1 nlnno 0 checksum 0x0 assoc 4 comdat 2
# CHECK-NEXT: [ 8](sec  4)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 assocSym
# CHECK-NEXT: [ 9](sec  5)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 secName
# CHECK-NEXT: AUX scnlen 0x4 nreloc 1 nlnno 0 checksum 0x0 assoc 5 comdat 2
# CHECK-NEXT: [11](sec  5)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 Symbol1
# CHECK-NEXT: [12](sec  6)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 secName
# CHECK-NEXT: AUX scnlen 0x4 nreloc 0 nlnno 0 checksum 0xb8bc6765 assoc 6 comdat 1
# CHECK-NEXT: [14](sec  6)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 Symbol2
# CHECK-NEXT: [15](sec  7)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 SecName
# CHECK-NEXT: AUX scnlen 0x4 nreloc 0 nlnno 0 checksum 0xb8bc6765 assoc 7 comdat 3
# CHECK-NEXT: [17](sec  7)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 Symbol3
# CHECK-NEXT: [18](sec  8)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 SecName
# CHECK-NEXT: AUX scnlen 0x4 nreloc 0 nlnno 0 checksum 0xb8bc6765 assoc 8 comdat 4
# CHECK-NEXT: [20](sec  8)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 Symbol4
# CHECK-NEXT: [21](sec 11)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 SecName
# CHECK-NEXT: AUX scnlen 0x4 nreloc 0 nlnno 0 checksum 0xb8bc6765 assoc 4 comdat 5
# CHECK-NEXT: [23](sec  9)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 SecName
# CHECK-NEXT: AUX scnlen 0x4 nreloc 0 nlnno 0 checksum 0xb8bc6765 assoc 9 comdat 6
# CHECK-NEXT: [25](sec  9)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 Symbol6
# CHECK-NEXT: [26](sec 10)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 SecName
# CHECK-NEXT: AUX scnlen 0x4 nreloc 0 nlnno 0 checksum 0xb8bc6765 assoc 10 comdat 7
# CHECK-NEXT: [28](sec 10)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 Symbol7
# CHECK-NEXT: [29](sec 12)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 assocSec
# CHECK-NEXT: AUX scnlen 0x4 nreloc 0 nlnno 0 checksum 0xb8bc6765 assoc 4 comdat 5
# CHECK-NEXT: [31](sec 11)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 Symbol5
# CHECK-NEXT: [32](sec 12)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 Symbol8
