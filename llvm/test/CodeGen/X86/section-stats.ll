; REQUIRES: asserts
; RUN: llc -o /dev/null -filetype=obj -stats %s 2>&1 | FileCheck %s

; CHECK-DAG: 1 elf-object-writer - Total size of SHF_ALLOC text sections
; CHECK-DAG: 1 elf-object-writer - Total size of SHF_ALLOC read-write sections
; CHECK-DAG: 512 elf-object-writer - Total size of section headers table
; CHECK-DAG: 64 elf-object-writer - Total size of ELF headers

target triple = "x86_64-unknown-linux-gnu"

@g = global i8 1

define void @f() {
    ret void
}
