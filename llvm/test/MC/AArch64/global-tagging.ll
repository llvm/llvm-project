;; Tagged symbols are only available on aarch64-linux-android.
; RUN: not llc %s -mtriple=aarch64-linux-unknown
; RUN: not llc %s -mtriple=x86_64-linux-unknown

; RUN: llc %s -mtriple=aarch64-linux-android31 -o %t.S
; RUN: FileCheck %s --input-file=%t.S --check-prefix=CHECK-ASM
; RUN: llvm-mc -filetype=obj %t.S -triple=aarch64-linux-android31 -o %t.o
; RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=CHECK-RELOCS

; RUN: obj2yaml %t.o -o %t.yaml
; RUN: FileCheck %s --input-file=%t.yaml --check-prefix=CHECK-YAML
; RUN: yaml2obj %t.yaml -o %t.o
; RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=CHECK-RELOCS

; CHECK-RELOCS:     Relocation section '.rela.memtag.globals.static' {{.*}} contains 4 entries
; CHECK-RELOCS:     R_AARCH64_NONE {{.*}} internal_four
; CHECK-RELOCS:     R_AARCH64_NONE {{.*}} four
; CHECK-RELOCS:     R_AARCH64_NONE {{.*}} sixteen
; CHECK-RELOCS:     R_AARCH64_NONE {{.*}} huge
; CHECK-RELOCS-NOT: specialcaselisted

; CHECK-YAML:      Sections:
; CHECK-YAML:      - Name: .rela.memtag.globals.static
; CHECK-YAML-NOT:  - Name:
; CHECK-YAML:      Relocations:
; CHECK-YAML-NEXT: - Symbol: internal_four
; CHECK-YAML-NEXT: Type: R_AARCH64_NONE
; CHECK-YAML-NEXT: - Symbol: four
; CHECK-YAML-NEXT: Type: R_AARCH64_NONE
; CHECK-YAML-NEXT: - Symbol: sixteen
; CHECK-YAML-NEXT: Type: R_AARCH64_NONE
; CHECK-YAML-NEXT: - Symbol: huge
; CHECK-YAML-NEXT: Type: R_AARCH64_NONE
; CHECK-YAML-NEXT: -

; CHECK-ASM: .memtag internal_four
; CHECK-ASM: .memtag four
; CHECK-ASM: .memtag sixteen
; CHECK-ASM: .memtag huge
; CHECK-ASM-NOT: .memtag specialcaselisted

@internal_four = internal global i32 1, sanitize_memtag
@four = global i32 1, sanitize_memtag
@sixteen = global [16 x i8] zeroinitializer, sanitize_memtag
@huge = global [16777232 x i8] zeroinitializer, sanitize_memtag
@specialcaselisted = global i16 2
