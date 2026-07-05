# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o %t
# RUN: lldb-test object-file %t | FileCheck %s

## Check that we are able to parse ELF files with more than SHN_LORESERVE
## sections. This generates a file that contains 64k sections from
## aaaaaaaa..dddddddd, plus a couple of standard ones (.strtab, etc.)
## Check the number is correct plus the names of a couple of chosen sections.

# CHECK: Showing 65540 sections
# CHECK: Name: aaaaaaaa
# CHECK: Name: bbbbbbbb
# CHECK: Name: cccccccc
# CHECK: Name: abcdabcd
# CHECK: Name: dddddddd

.macro gen_sections4 x
  .section a\x
  .section b\x
  .section c\x
  .section d\x
.endm

.macro gen_sections16 x
  gen_sections4 a\x
  gen_sections4 b\x
  gen_sections4 c\x
  gen_sections4 d\x
.endm

.macro gen_sections64 x
  gen_sections16 a\x
  gen_sections16 b\x
  gen_sections16 c\x
  gen_sections16 d\x
.endm

.macro gen_sections256 x
  gen_sections64 a\x
  gen_sections64 b\x
  gen_sections64 c\x
  gen_sections64 d\x
.endm

.macro gen_sections1024 x
  gen_sections256 a\x
  gen_sections256 b\x
  gen_sections256 c\x
  gen_sections256 d\x
.endm

.macro gen_sections4096 x
  gen_sections1024 a\x
  gen_sections1024 b\x
  gen_sections1024 c\x
  gen_sections1024 d\x
.endm

.macro gen_sections16384 x
  gen_sections4096 a\x
  gen_sections4096 b\x
  gen_sections4096 c\x
  gen_sections4096 d\x
.endm

gen_sections16384 a
gen_sections16384 b
gen_sections16384 c
gen_sections16384 d

.global _start
_start:
