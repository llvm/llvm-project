; REQUIRES: asserts
; RUN: llc -o /dev/null -filetype=obj -stats %s 2>&1 | FileCheck %s

; CHECK-DAG: 1 elf-object-writer - Total size of SHF_ALLOC text sections
; CHECK-DAG: 1 elf-object-writer - Total size of SHF_ALLOC read-write sections

target triple = "x86_64-unknown-linux-gnu"

@g = global i8 1

define void @f() {
    ret void
}
