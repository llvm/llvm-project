# REQUIRES: x86-registered-target

## It is possible for the section header table and symbol table to share the
## same string table for storing section and symbol names. This test shows that
## under various circumstances, the names are still correct after llvm-objcopy
## has copied such an object file, and that the name table is still shared.
## This test uses the assembler rather than yaml2obj because yaml2obj generates
## separate string tables, whereas the assembler shares them.

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o %t.o
## Sanity check that the string tables are shared:
# RUN: llvm-readobj --section-headers %t.o \
# RUN:   | FileCheck %s --check-prefix=VALIDATE --implicit-check-not=.shstrtab

# VALIDATE: Name: .strtab

## Case 1: basic copy.
# RUN: llvm-objcopy %t.o %t.basic
# RUN: llvm-readobj --section-headers --symbols %t.basic \
# RUN:   | FileCheck %s --check-prefix=BASIC --implicit-check-not=.shstrtab

# BASIC: Sections [
# BASIC:   Name: .foo (
# BASIC:   Name: .strtab (
# BASIC: Symbols [
# BASIC:   Name: foo (

## Case 2: renaming a section.
# RUN: llvm-objcopy %t.o %t.rename-section --rename-section .foo=.oof
# RUN: llvm-readobj --section-headers --symbols %t.rename-section \
# RUN:   | FileCheck %s --check-prefix=SECTION-RENAME --implicit-check-not=.shstrtab

# SECTION-RENAME: Sections [
# SECTION-RENAME:   Name: .oof (
# SECTION-RENAME:   Name: .strtab (
# SECTION-RENAME: Symbols [
# SECTION-RENAME:   Name: foo (

## Case 3: renaming a symbol.
# RUN: llvm-objcopy %t.o %t.redefine-symbol --redefine-sym foo=oof
# RUN: llvm-readobj --section-headers --symbols %t.redefine-symbol \
# RUN:   | FileCheck %s --check-prefix=SYMBOL-RENAME --implicit-check-not=.shstrtab

# SYMBOL-RENAME: Sections [
# SYMBOL-RENAME:   Name: .foo (
# SYMBOL-RENAME:   Name: .strtab (
# SYMBOL-RENAME: Symbols [
# SYMBOL-RENAME:   Name: oof (

## Case 4: removing a section.
# RUN: llvm-objcopy %t.o %t.remove-section -R .foo
# RUN: llvm-readobj --section-headers --symbols %t.remove-section \
# RUN:   | FileCheck %s --check-prefix=SECTION-REMOVE --implicit-check-not=.shstrtab --implicit-check-not=.foo

# SECTION-REMOVE: Sections [
# SECTION-REMOVE:   Name: .strtab (
# SECTION-REMOVE: Symbols [
# SECTION-REMOVE:   Name: foo (

## Case 5: removing a symbol.
# RUN: llvm-objcopy %t.o %t.remove-symbol -N foo
# RUN: llvm-readobj --section-headers --symbols %t.remove-symbol \
# RUN:   | FileCheck %s --check-prefix=SYMBOL-REMOVE --implicit-check-not=.shstrtab --implicit-check-not=foo

# SYMBOL-REMOVE: Sections [
# SYMBOL-REMOVE:   Name: .foo (
# SYMBOL-REMOVE:   Name: .strtab (
# SYMBOL-REMOVE: Symbols [

## Case 6: adding a section.
# RUN: llvm-objcopy %t.o %t.add-section --add-section .bar=%s
# RUN: llvm-readobj --section-headers --symbols %t.add-section \
# RUN:   | FileCheck %s --check-prefix=SECTION-ADD --implicit-check-not=.shstrtab

# SECTION-ADD: Sections [
# SECTION-ADD:   Name: .foo (
# SECTION-ADD:   Name: .strtab (
# SECTION-ADD:   Name: .bar (
# SECTION-ADD: Symbols [
# SECTION-ADD:   Name: foo (

## Case 7: adding a symbol.
# RUN: llvm-objcopy %t.o %t.add-symbol --add-symbol bar=0x1234
# RUN: llvm-readobj --section-headers --symbols %t.add-symbol \
# RUN:   | FileCheck %s --check-prefix=SYMBOL-ADD --implicit-check-not=.shstrtab

# SYMBOL-ADD: Sections [
# SYMBOL-ADD:   Name: .foo (
# SYMBOL-ADD:   Name: .strtab (
# SYMBOL-ADD: Symbols [
# SYMBOL-ADD:   Name: foo (
# SYMBOL-ADD:   Name: bar (

## Case 8: removing all symbols.
# RUN: llvm-objcopy %t.o %t.strip-all --strip-all
# RUN: llvm-readobj --section-headers --symbols %t.strip-all \
# RUN:   | FileCheck %s --check-prefix=STRIP-ALL --implicit-check-not=.shstrtab

# STRIP-ALL:      Sections [
# STRIP-ALL:        Name: .foo (
# STRIP-ALL:        Name: .strtab (
# STRIP-ALL:      Symbols [
# STRIP-ALL-NEXT: ]

.section .foo,"a",@progbits
foo = 0x4321
