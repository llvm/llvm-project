;; Tagged symbols are only available on aarch64-linux-android.
; RUN: not llc -filetype=null %s -mtriple=aarch64-unknown-linux 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: %if x86-registered-target %{ not llc -filetype=null %s -mtriple=x86_64-unknown-linux 2>&1 | FileCheck %s --check-prefix=ERR %}

; ERR: error: tagged symbols (-fsanitize=memtag-globals) are only supported on AArch64 Android

; RUN: llc %s -mtriple=aarch64-linux-android31 -o %t.S
; RUN: FileCheck %s --input-file=%t.S --check-prefix=CHECK-ASM
; RUN: llvm-mc -filetype=obj %t.S -triple=aarch64-linux-android31 -o %t.o
; RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=CHECK-RELOCS

; RUN: obj2yaml %t.o -o %t.o.yaml
; RUN: FileCheck %s --input-file=%t.o.yaml --check-prefix=CHECK-OYAML
; RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=CHECK-RELOCS
; RUN: ld.lld %t.o -o %t.so
; RUN: obj2yaml %t.so -o %t.so.yaml
; RUN: FileCheck %s --input-file=%t.so.yaml --check-prefix=CHECK-SOYAML

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

; CHECK-OYAML:      Sections:
; CHECK-OYAML:      - Name: .rela.memtag.globals.static
; CHECK-OYAML-NOT:  - Name:
; CHECK-OYAML:      Relocations:
; CHECK-OYAML-NEXT: - Symbol: internal_four
; CHECK-OYAML-NEXT: Type: R_AARCH64_NONE
; CHECK-OYAML-NEXT: - Symbol: four
; CHECK-OYAML-NEXT: Type: R_AARCH64_NONE
; CHECK-OYAML-NEXT: - Symbol: sixteen
; CHECK-OYAML-NEXT: Type: R_AARCH64_NONE
; CHECK-OYAML-NEXT: - Symbol: huge
; CHECK-OYAML-NEXT: Type: R_AARCH64_NONE
; CHECK-OYAML-NEXT: -

;; Value: 0x{{.*}}0 checks for 16-alignment of address
; CHECK-SOYAML: Symbols:
; CHECK-SOYAML:  - Name: internal_four
; CHECK-SOYAML:    Value: 0x{{.*}}0{{$}}
; CHECK-SOYAML:  - Name: four
; CHECK-SOYAML:    Value: 0x{{.*}}0{{$}}
; CHECK-SOYAML:  - Name: sixteen
; CHECK-SOYAML:    Value: 0x{{.*}}0{{$}}
; CHECK-SOYAML:  - Name: huge
; CHECK-SOYAML:    Value: 0x{{.*}}0{{$}}
;; At least as currently laid out, specialcaselisted gets put adjacient to a
;; tagged global, so it also has to be aligned to the next granule.
; CHECK-SOYAML:  - Name: specialcaselisted
; CHECK-SOYAML:    Value: 0x{{.*}}0{{$}}

; CHECK-ASM: .memtag internal_four
; CHECK-ASM .p2align        4
; CHECK-ASM: .size   internal_four, 16
; CHECK-ASM: .memtag four
; CHECK-ASM .p2align        4
; CHECK-ASM: .size   four, 16
; CHECK-ASM: .memtag sixteen
; CHECK-ASM .p2align        4
; CHECK-ASM: .size   sixteen, 16
; CHECK-ASM: .memtag huge
; CHECK-ASM .p2align        4
; CHECK-ASM: .size   huge, 16777232
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
