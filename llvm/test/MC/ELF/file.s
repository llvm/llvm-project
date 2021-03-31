# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o - | llvm-readelf -s - | FileCheck %s

# CHECK:       0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
# CHECK-NEXT:  1: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS foo.c
# CHECK-NEXT:  2: 0000000000000000     0 SECTION LOCAL  DEFAULT     2 .text
# CHECK-NEXT:  3: 0000000000000000     0 SECTION LOCAL  DEFAULT     4 foo
# CHECK-NEXT:  4: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     4 local0
# CHECK-NEXT:  5: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS bar.c
# CHECK-NEXT:  6: 0000000000000000     0 SECTION LOCAL  DEFAULT     6 bar0
# CHECK-NEXT:  7: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     6 local1
# CHECK-NEXT:  8: 0000000000000000     0 SECTION LOCAL  DEFAULT     8 bar1
# CHECK-NEXT:  9: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     8 local2
# CHECK-NEXT: 10: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS bar.c
# CHECK-NEXT: 11: 0000000000000008     0 NOTYPE  GLOBAL DEFAULT     2 foo.c
# CHECK-NEXT: 12: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT     6 bar.c

.quad .text

## A STT_FILE symbol and a symbol of the same name can coexist.
.file "foo.c"
.globl foo.c
foo.c:
.section foo,"a"
local0:
.quad foo

## STT_FILE "bar.c" precedes subsequently defined local symbols.
.file "bar.c"
.section bar0,"a"
.globl bar.c
bar.c:
local1:
.quad bar0

.section bar1,"a"
local2:
.quad bar1

## STT_FILE symbols of the same name are not de-duplicated.
.file "bar.c"
