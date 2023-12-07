;; Tagged symbols are only available on aarch64-linux-android.
; RUN: not llc -filetype=null %s -mtriple=aarch64-unknown-linux 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: %if x86-registered-target %{ not llc -filetype=null %s -mtriple=x86_64-unknown-linux 2>&1 | FileCheck %s --check-prefix=ERR %}

; ERR: error: tagged symbols (-fsanitize=memtag-globals) are only supported on AArch64 Android

; RUN: llc %s -mtriple=aarch64-linux-android31 -o %t.S
; RUN: FileCheck %s --input-file=%t.S --check-prefix=CHECK-ASM
; RUN: llvm-mc -filetype=obj %t.S -triple=aarch64-linux-android31 -o %t.o
; RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=CHECK-RELOCS

; RUN: obj2yaml %t.o -o %t.yaml
; RUN: FileCheck %s --input-file=%t.yaml --check-prefix=CHECK-YAML
; RUN: yaml2obj %t.yaml -o %t.o
; RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=CHECK-RELOCS

;; Check we don't create relocations referencing a section symbol for sanitize_memtag globals.
; CHECK-RELOCS:      Relocation section '.rela.text' {{.*}} contains 4 entries:
; CHECK-RELOCS:      R_AARCH64_ADR_GOT_PAGE     {{.*}} internal_four + 0
; CHECK-RELOCS-NEXT: R_AARCH64_ADR_GOT_PAGE     {{.*}} four + 0
; CHECK-RELOCS-NEXT: R_AARCH64_LD64_GOT_LO12_NC {{.*}} internal_four + 0
; CHECK-RELOCS-NEXT: R_AARCH64_LD64_GOT_LO12_NC {{.*}} four + 0

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
@four = dso_local global i32 1, sanitize_memtag
@sixteen = global [16 x i8] zeroinitializer, sanitize_memtag
@huge = global [16777232 x i8] zeroinitializer, sanitize_memtag
@specialcaselisted = global i16 2

define i32 @use() {
entry:
  %a = load i32, ptr @internal_four
  %b = load i32, ptr @four
  %sum = add i32 %a, %b
  ret i32 %sum
}
